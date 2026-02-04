# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dynamic N-gram Proposer for CPU-GPU Hybrid Speculative Decoding.

This module extends the base N-gram proposer with dynamic learning capabilities:
- Learns patterns from generated outputs in real-time
- Maintains an N-gram frequency table for better predictions
- Runs entirely on CPU to offload work from GPU
- Thread-safe for concurrent access

Key features over base NgramProposer:
1. Dynamic learning from output tokens (not just prompt lookup)
2. Frequency-based prediction confidence
3. Background update thread
4. Statistics tracking for acceptance rate analysis
"""

import queue
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

logger = init_logger(__name__)


@dataclass
class DynamicNgramConfig:
    """Configuration for dynamic N-gram proposer."""

    min_n: int = 2
    """Minimum N-gram size."""

    max_n: int = 4
    """Maximum N-gram size."""

    num_speculative_tokens: int = 5
    """Number of tokens to propose."""

    min_frequency: int = 2
    """Minimum frequency threshold for proposals."""

    max_table_size: int = 100000
    """Maximum N-gram table size (LRU eviction after this)."""

    enable_background_update: bool = True
    """Enable background thread for table updates."""

    decay_factor: float = 0.99
    """Frequency decay factor for aging patterns."""


@dataclass
class NgramStats:
    """Statistics for N-gram proposer performance."""

    total_proposals: int = 0
    """Total number of proposal attempts."""

    successful_proposals: int = 0
    """Proposals that returned tokens."""

    accepted_tokens: int = 0
    """Number of accepted speculative tokens."""

    rejected_tokens: int = 0
    """Number of rejected speculative tokens."""

    table_size: int = 0
    """Current N-gram table size."""

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate."""
        total = self.accepted_tokens + self.rejected_tokens
        return self.accepted_tokens / total if total > 0 else 0.0

    @property
    def proposal_hit_rate(self) -> float:
        """Calculate proposal hit rate."""
        return self.successful_proposals / self.total_proposals if self.total_proposals > 0 else 0.0


class DynamicNgramProposer:
    """
    Dynamic N-gram Proposer with learning capabilities.

    This proposer learns from generated outputs and maintains a frequency
    table of N-grams for better prediction accuracy. It runs entirely on
    CPU to offload computation from GPU.
    """

    def __init__(
        self,
        config: Optional[DynamicNgramConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
    ):
        """
        Initialize the dynamic N-gram proposer.

        Args:
            config: Dynamic N-gram configuration.
            vllm_config: vLLM configuration (for compatibility with base proposer).
        """
        if config is None:
            config = DynamicNgramConfig()

        self.config = config
        self.min_n = config.min_n
        self.max_n = config.max_n
        self.k = config.num_speculative_tokens
        self.min_frequency = config.min_frequency

        # N-gram frequency table: (n-1)-gram -> {next_token: count}
        # Using tuple keys for hashability
        self._ngram_table: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Lock for thread-safe access
        self._lock = threading.RLock()

        # Background update queue and thread
        self._update_queue: queue.Queue = queue.Queue()
        self._shutdown = threading.Event()

        if config.enable_background_update:
            self._update_thread = threading.Thread(
                target=self._background_update_loop,
                daemon=True,
                name="ngram-update-thread"
            )
            self._update_thread.start()
        else:
            self._update_thread = None

        # Statistics
        self.stats = NgramStats()

        # Fallback to base proposer for prompt lookup
        self._base_proposer: Optional[NgramProposer] = None
        if vllm_config is not None:
            try:
                self._base_proposer = NgramProposer(vllm_config)
            except Exception as e:
                logger.warning(f"Failed to initialize base NgramProposer: {e}")

        logger.info(
            f"DynamicNgramProposer initialized: n={self.min_n}-{self.max_n}, "
            f"k={self.k}, min_freq={self.min_frequency}"
        )

    def propose(
        self,
        context_token_ids: np.ndarray,
        use_dynamic_table: bool = True,
        use_prompt_lookup: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Propose speculative tokens based on N-gram patterns.

        Args:
            context_token_ids: Context token IDs as numpy array.
            use_dynamic_table: Use learned N-gram table for proposals.
            use_prompt_lookup: Fall back to prompt lookup if table miss.

        Returns:
            Array of proposed token IDs, or None if no proposal.
        """
        self.stats.total_proposals += 1

        # Try dynamic table first
        if use_dynamic_table:
            proposals = self._propose_from_table(context_token_ids)
            if proposals is not None and len(proposals) > 0:
                self.stats.successful_proposals += 1
                return proposals

        # Fall back to prompt lookup
        if use_prompt_lookup and self._base_proposer is not None:
            proposals = self._base_proposer.propose(context_token_ids)
            if proposals is not None:
                self.stats.successful_proposals += 1
                return proposals

        return None

    def _propose_from_table(
        self,
        context_token_ids: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Propose tokens from learned N-gram table."""
        proposals = []
        current_context = list(context_token_ids)

        with self._lock:
            for _ in range(self.k):
                # Try different N-gram sizes (larger first for specificity)
                next_token = None
                best_confidence = 0

                for n in range(self.max_n, self.min_n - 1, -1):
                    if len(current_context) < n - 1:
                        continue

                    prefix = tuple(current_context[-(n - 1):])

                    if prefix in self._ngram_table:
                        candidates = self._ngram_table[prefix]

                        # Find most frequent next token
                        if candidates:
                            best_token, count = max(
                                candidates.items(),
                                key=lambda x: x[1]
                            )

                            if count >= self.min_frequency:
                                total = sum(candidates.values())
                                confidence = count / total

                                if confidence > best_confidence:
                                    next_token = best_token
                                    best_confidence = confidence

                if next_token is not None:
                    proposals.append(next_token)
                    current_context.append(next_token)
                else:
                    # No more predictions possible
                    break

        if proposals:
            return np.array(proposals, dtype=np.int32)
        return None

    def update(self, token_ids: List[int]):
        """
        Update N-gram table with new tokens.

        Args:
            token_ids: List of token IDs to learn from.
        """
        if self.config.enable_background_update:
            # Queue for background processing
            self._update_queue.put(token_ids)
        else:
            # Direct update
            self._update_table(token_ids)

    def _update_table(self, token_ids: List[int]):
        """Update N-gram table with token sequence."""
        if len(token_ids) < self.min_n:
            return

        with self._lock:
            # Add N-grams of all sizes
            for n in range(self.min_n, self.max_n + 1):
                for i in range(len(token_ids) - n + 1):
                    prefix = tuple(token_ids[i:i + n - 1])
                    next_token = token_ids[i + n - 1]
                    self._ngram_table[prefix][next_token] += 1

            # Update stats
            self.stats.table_size = len(self._ngram_table)

            # Check for table size limit
            if self.stats.table_size > self.config.max_table_size:
                self._evict_old_entries()

    def _evict_old_entries(self):
        """Evict old entries using frequency decay."""
        # Apply decay to all entries
        entries_to_remove = []

        for prefix, candidates in self._ngram_table.items():
            for token, count in list(candidates.items()):
                new_count = int(count * self.config.decay_factor)
                if new_count < 1:
                    del candidates[token]
                else:
                    candidates[token] = new_count

            if not candidates:
                entries_to_remove.append(prefix)

        for prefix in entries_to_remove:
            del self._ngram_table[prefix]

        self.stats.table_size = len(self._ngram_table)
        logger.debug(f"N-gram table eviction: {len(entries_to_remove)} prefixes removed")

    def _background_update_loop(self):
        """Background thread for N-gram table updates."""
        while not self._shutdown.is_set():
            try:
                token_ids = self._update_queue.get(timeout=0.1)
                self._update_table(token_ids)
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Background update error: {e}")

    def record_acceptance(self, num_accepted: int, num_rejected: int):
        """
        Record acceptance/rejection statistics.

        Args:
            num_accepted: Number of accepted speculative tokens.
            num_rejected: Number of rejected speculative tokens.
        """
        self.stats.accepted_tokens += num_accepted
        self.stats.rejected_tokens += num_rejected

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "total_proposals": self.stats.total_proposals,
            "successful_proposals": self.stats.successful_proposals,
            "accepted_tokens": self.stats.accepted_tokens,
            "rejected_tokens": self.stats.rejected_tokens,
            "acceptance_rate": self.stats.acceptance_rate,
            "proposal_hit_rate": self.stats.proposal_hit_rate,
            "table_size": self.stats.table_size,
        }

    def shutdown(self):
        """Shutdown the proposer and background thread."""
        self._shutdown.set()
        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)

    def load_model(self, *args, **kwargs):
        """No model to load (CPU-only N-gram matching)."""
        pass


class HybridNgramWorker:
    """
    N-gram worker for CPU-GPU hybrid speculative decoding.

    This worker manages the N-gram proposer and coordinates with
    the GPU model for verification.
    """

    def __init__(
        self,
        config: DynamicNgramConfig,
        vllm_config: Optional[VllmConfig] = None,
    ):
        """
        Initialize the hybrid N-gram worker.

        Args:
            config: N-gram configuration.
            vllm_config: vLLM configuration.
        """
        self.proposer = DynamicNgramProposer(config, vllm_config)
        self.config = config

    def get_proposals(
        self,
        context_token_ids: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Get speculative token proposals (runs on CPU).

        Args:
            context_token_ids: Context token IDs.

        Returns:
            Proposed token IDs or None.
        """
        return self.proposer.propose(context_token_ids)

    def submit_output(self, output_token_ids: List[int]):
        """
        Submit generated output for learning.

        Args:
            output_token_ids: Generated token IDs.
        """
        self.proposer.update(output_token_ids)

    def record_verification_result(
        self,
        proposed_tokens: np.ndarray,
        accepted_mask: np.ndarray,
    ):
        """
        Record verification results for statistics.

        Args:
            proposed_tokens: Proposed token IDs.
            accepted_mask: Boolean mask of accepted tokens.
        """
        num_accepted = accepted_mask.sum()
        num_rejected = len(accepted_mask) - num_accepted
        self.proposer.record_acceptance(num_accepted, num_rejected)

    def get_stats(self) -> Dict:
        """Get worker statistics."""
        return self.proposer.get_stats()

    def shutdown(self):
        """Shutdown the worker."""
        self.proposer.shutdown()
