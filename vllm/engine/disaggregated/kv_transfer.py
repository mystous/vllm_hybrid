# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache Transfer for Disaggregated Serving.

This module implements KV cache transfer between Prefill and Decode nodes
with multiple transport backends:
- TCP: Standard network transfer
- SHM: Shared memory for same-machine deployment
- RDMA: High-performance RDMA (requires NCCL/Gloo with RDMA support)

Key features:
- Async transfer with pipelining
- Compression support for bandwidth reduction
- Request ID based routing
"""

import asyncio
import pickle
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class TransferMethod(str, Enum):
    """KV cache transfer method."""
    TCP = "tcp"
    SHM = "shm"
    RDMA = "rdma"
    GLOO = "gloo"


@dataclass
class KVTransferConfig:
    """Configuration for KV cache transfer."""

    method: TransferMethod = TransferMethod.TCP
    """Transfer method: tcp, shm, rdma, or gloo."""

    host: str = "localhost"
    """Target host for TCP/RDMA."""

    port: int = 29500
    """Target port for TCP/RDMA."""

    shm_path: str = "/dev/shm/vllm_kv_cache"
    """Shared memory path for SHM method."""

    buffer_size_mb: int = 1024
    """Buffer size in MB for transfers."""

    compress: bool = False
    """Enable compression for transfers."""

    compression_level: int = 1
    """Compression level (1-9, higher = better ratio but slower)."""

    timeout_seconds: float = 30.0
    """Transfer timeout in seconds."""

    max_pending_transfers: int = 16
    """Maximum pending async transfers."""


@dataclass
class KVCacheData:
    """KV cache data container for transfer."""

    request_id: str
    """Request ID for routing."""

    key_cache: List[torch.Tensor]
    """Key cache tensors per layer."""

    value_cache: List[torch.Tensor]
    """Value cache tensors per layer."""

    num_tokens: int
    """Number of tokens in the cache."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


class KVTransferBackend(ABC):
    """Abstract base class for KV transfer backends."""

    @abstractmethod
    async def send(self, data: KVCacheData) -> bool:
        """Send KV cache data."""
        pass

    @abstractmethod
    async def receive(self, request_id: str, timeout: float) -> Optional[KVCacheData]:
        """Receive KV cache data."""
        pass

    @abstractmethod
    def close(self):
        """Close the backend."""
        pass


class TCPTransferBackend(KVTransferBackend):
    """TCP-based KV cache transfer."""

    def __init__(self, config: KVTransferConfig, is_sender: bool):
        self.config = config
        self.is_sender = is_sender
        self._server = None
        self._connections: Dict[str, asyncio.StreamWriter] = {}
        self._pending_data: Dict[str, KVCacheData] = {}
        self._lock = asyncio.Lock()

    async def start_server(self):
        """Start TCP server for receiving."""
        if not self.is_sender:
            self._server = await asyncio.start_server(
                self._handle_connection,
                self.config.host,
                self.config.port,
            )
            logger.info(f"KV Transfer server started on {self.config.host}:{self.config.port}")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle incoming connection."""
        try:
            while True:
                # Read message length (4 bytes)
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, 'big')

                # Read message
                data_bytes = await reader.readexactly(length)

                # Decompress if needed
                if self.config.compress:
                    import zlib
                    data_bytes = zlib.decompress(data_bytes)

                # Deserialize
                kv_data = self._deserialize(data_bytes)

                # Store for retrieval
                async with self._lock:
                    self._pending_data[kv_data.request_id] = kv_data

        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            logger.error(f"Connection handler error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def send(self, data: KVCacheData) -> bool:
        """Send KV cache data via TCP."""
        try:
            # Serialize
            data_bytes = self._serialize(data)

            # Compress if enabled
            if self.config.compress:
                import zlib
                data_bytes = zlib.compress(data_bytes, level=self.config.compression_level)

            # Connect if needed
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.host, self.config.port),
                timeout=self.config.timeout_seconds,
            )

            # Send length + data
            length = len(data_bytes)
            writer.write(length.to_bytes(4, 'big'))
            writer.write(data_bytes)
            await writer.drain()

            writer.close()
            await writer.wait_closed()

            logger.debug(f"Sent KV cache for request {data.request_id}, {length} bytes")
            return True

        except Exception as e:
            logger.error(f"TCP send error: {e}")
            return False

    async def receive(self, request_id: str, timeout: float) -> Optional[KVCacheData]:
        """Receive KV cache data."""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            async with self._lock:
                if request_id in self._pending_data:
                    return self._pending_data.pop(request_id)

            await asyncio.sleep(0.01)  # Small sleep to avoid busy waiting

        return None

    def _serialize(self, data: KVCacheData) -> bytes:
        """Serialize KV cache data."""
        # Move tensors to CPU and convert to numpy for serialization
        serializable = {
            "request_id": data.request_id,
            "key_cache": [t.cpu().numpy() for t in data.key_cache],
            "value_cache": [t.cpu().numpy() for t in data.value_cache],
            "num_tokens": data.num_tokens,
            "metadata": data.metadata,
        }
        return pickle.dumps(serializable)

    def _deserialize(self, data_bytes: bytes) -> KVCacheData:
        """Deserialize KV cache data."""
        import numpy as np
        serializable = pickle.loads(data_bytes)
        return KVCacheData(
            request_id=serializable["request_id"],
            key_cache=[torch.from_numpy(arr) for arr in serializable["key_cache"]],
            value_cache=[torch.from_numpy(arr) for arr in serializable["value_cache"]],
            num_tokens=serializable["num_tokens"],
            metadata=serializable["metadata"],
        )

    def close(self):
        """Close the backend."""
        if self._server is not None:
            self._server.close()


class SharedMemoryBackend(KVTransferBackend):
    """Shared memory based KV cache transfer for same-machine deployment."""

    def __init__(self, config: KVTransferConfig, is_sender: bool):
        self.config = config
        self.is_sender = is_sender
        self._shm_path = config.shm_path
        self._lock = threading.Lock()
        self._pending: Dict[str, KVCacheData] = {}

    async def send(self, data: KVCacheData) -> bool:
        """Send via shared memory."""
        try:
            import os
            import tempfile

            # Create a temporary file for this request
            shm_file = f"{self._shm_path}_{data.request_id}"

            # Serialize
            serialized = self._serialize_tensors(data)

            # Write to shared memory
            with open(shm_file, 'wb') as f:
                pickle.dump(serialized, f)

            logger.debug(f"Wrote KV cache to SHM: {shm_file}")
            return True

        except Exception as e:
            logger.error(f"SHM send error: {e}")
            return False

    async def receive(self, request_id: str, timeout: float) -> Optional[KVCacheData]:
        """Receive from shared memory."""
        import os

        shm_file = f"{self._shm_path}_{request_id}"
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if os.path.exists(shm_file):
                try:
                    with open(shm_file, 'rb') as f:
                        serialized = pickle.load(f)

                    # Clean up
                    os.remove(shm_file)

                    return self._deserialize_tensors(serialized)

                except Exception as e:
                    logger.error(f"SHM receive error: {e}")
                    return None

            await asyncio.sleep(0.01)

        return None

    def _serialize_tensors(self, data: KVCacheData) -> Dict:
        """Serialize tensors for SHM."""
        return {
            "request_id": data.request_id,
            "key_cache": [t.cpu() for t in data.key_cache],
            "value_cache": [t.cpu() for t in data.value_cache],
            "num_tokens": data.num_tokens,
            "metadata": data.metadata,
        }

    def _deserialize_tensors(self, serialized: Dict) -> KVCacheData:
        """Deserialize tensors from SHM."""
        return KVCacheData(
            request_id=serialized["request_id"],
            key_cache=serialized["key_cache"],
            value_cache=serialized["value_cache"],
            num_tokens=serialized["num_tokens"],
            metadata=serialized["metadata"],
        )

    def close(self):
        """Clean up shared memory files."""
        import glob
        import os

        for f in glob.glob(f"{self._shm_path}_*"):
            try:
                os.remove(f)
            except OSError:
                pass


class KVCacheSender:
    """
    KV Cache sender for Prefill nodes.

    Handles sending KV cache to Decode nodes after prefill completion.
    """

    def __init__(self, config: KVTransferConfig):
        """
        Initialize KV cache sender.

        Args:
            config: Transfer configuration.
        """
        self.config = config

        # Initialize backend
        if config.method == TransferMethod.TCP:
            self._backend = TCPTransferBackend(config, is_sender=True)
        elif config.method == TransferMethod.SHM:
            self._backend = SharedMemoryBackend(config, is_sender=True)
        else:
            raise ValueError(f"Unsupported transfer method: {config.method}")

        # Async transfer tracking
        self._pending_transfers: Dict[str, asyncio.Task] = {}
        self._transfer_semaphore = asyncio.Semaphore(config.max_pending_transfers)

        logger.info(f"KVCacheSender initialized with {config.method.value} backend")

    async def send_async(
        self,
        kv_cache: Tuple[List[torch.Tensor], List[torch.Tensor]],
        request_id: str,
        num_tokens: int,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Send KV cache asynchronously.

        Args:
            kv_cache: Tuple of (key_cache, value_cache) tensor lists.
            request_id: Request ID for routing.
            num_tokens: Number of tokens in the cache.
            metadata: Optional metadata.

        Returns:
            Transfer handle (request_id).
        """
        key_cache, value_cache = kv_cache

        data = KVCacheData(
            request_id=request_id,
            key_cache=key_cache,
            value_cache=value_cache,
            num_tokens=num_tokens,
            metadata=metadata or {},
        )

        # Acquire semaphore for rate limiting
        async with self._transfer_semaphore:
            task = asyncio.create_task(self._backend.send(data))
            self._pending_transfers[request_id] = task

        return request_id

    async def wait_for_completion(self, request_id: str) -> bool:
        """Wait for a specific transfer to complete."""
        if request_id in self._pending_transfers:
            return await self._pending_transfers[request_id]
        return True

    def close(self):
        """Close the sender."""
        self._backend.close()


class KVCacheReceiver:
    """
    KV Cache receiver for Decode nodes.

    Handles receiving KV cache from Prefill nodes.
    """

    def __init__(self, config: KVTransferConfig):
        """
        Initialize KV cache receiver.

        Args:
            config: Transfer configuration.
        """
        self.config = config

        # Initialize backend
        if config.method == TransferMethod.TCP:
            self._backend = TCPTransferBackend(config, is_sender=False)
        elif config.method == TransferMethod.SHM:
            self._backend = SharedMemoryBackend(config, is_sender=False)
        else:
            raise ValueError(f"Unsupported transfer method: {config.method}")

        logger.info(f"KVCacheReceiver initialized with {config.method.value} backend")

    async def start(self):
        """Start the receiver (for TCP server)."""
        if isinstance(self._backend, TCPTransferBackend):
            await self._backend.start_server()

    async def receive_async(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor], int]]:
        """
        Receive KV cache asynchronously.

        Args:
            request_id: Request ID to receive.
            timeout: Timeout in seconds (uses config default if None).

        Returns:
            Tuple of (key_cache, value_cache, num_tokens) or None if timeout.
        """
        timeout = timeout or self.config.timeout_seconds

        data = await self._backend.receive(request_id, timeout)

        if data is not None:
            return (data.key_cache, data.value_cache, data.num_tokens)

        return None

    def close(self):
        """Close the receiver."""
        self._backend.close()
