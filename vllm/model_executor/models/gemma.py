# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Copyright (c) Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model compatible with HuggingFace weights."""

from collections.abc import Iterable
from functools import cache
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import GemmaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


@cache
def _get_gemma_act_fn(
    hidden_act: str | None,
    hidden_activation: str | None,
) -> nn.Module:
    if hidden_activation is None:
        if hidden_act is not None:
            logger.warning(
                "Gemma's activation function was incorrectly set to exact GeLU "
                "in the config JSON file when it was initially released. "
                "Changing the activation function to approximate GeLU "
                "(`gelu_pytorch_tanh`). If you want to use the legacy "
                "`%s`, edit the config JSON to set "
                "`hidden_activation=%s` instead of `hidden_act`. "
                "See https://github.com/huggingface/transformers/pull/29402 "
                "for more details.",
                hidden_act,
                hidden_act,
            )
        return GeluAndMul(approximate="tanh")
    elif hidden_activation == "gelu_pytorch_tanh":
        return GeluAndMul(approximate="tanh")
    elif hidden_activation == "gelu":
        return GeluAndMul(approximate="none")
    else:
        raise ValueError(
            f"Activation function {hidden_act} is not supported for Gemma models."
        )


class GemmaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str | None = None,
        hidden_activation: str | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = _get_gemma_act_fn(hidden_act, hidden_activation)

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GemmaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class GemmaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = GemmaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_activation=getattr(config, "hidden_activation", None),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    # ------------------------------------------------------------------
    # TSK_016 Step 5.6 — NEO sub-batch stage helpers (mirrors the Llama
    # pattern). Vanilla forward path above is untouched; these helpers
    # are dead code unless ``enable_neo_asymmetric`` is True.
    # ------------------------------------------------------------------
    def neo_preproj(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def neo_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.self_attn(positions=positions, hidden_states=hidden_states)

    def neo_postproj(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class GemmaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: GemmaDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer), persistent=False)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # ------------------------------------------------------------------
    # TSK_016 Step 5.6 — NEO asymmetric pipelined forward.
    # Mirrors ``LlamaModel.forward_neo_pipelined`` (vllm/model_executor/
    # models/llama.py) — see the Llama version for the algorithm rationale
    # (layer-offset ping-pong + per-sub-batch ForwardContext fork).
    # ------------------------------------------------------------------
    def forward_neo_pipelined(
        self,
        sub_batches,                                   # list[SubBatch]
        positions_list: list[torch.Tensor],            # one per sub-batch
        embeddings_list: list[torch.Tensor],           # one per sub-batch
        per_subbatch_contexts=None,                    # list[ForwardContext] | None
    ):
        from vllm.forward_context import override_forward_context
        from vllm.v1.worker.sub_batch_executor import (
            LayerPipelineCallbacks,
            SubBatchPipelineExecutor,
        )

        if len(sub_batches) not in (1, 2):
            raise ValueError(
                f"forward_neo_pipelined expects 1 or 2 sub-batches, "
                f"got {len(sub_batches)}"
            )

        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        num_layers = len(layers)
        residuals: dict[tuple[int, int], torch.Tensor | None] = {}
        _bid_to_ubid: dict[int, int] = {
            id(sb): ubid for ubid, sb in enumerate(sub_batches)
        }

        def _bid(batch) -> int:
            return id(batch)

        def preproj(emb, batch, layer_idx, layer_off):
            del layer_off
            layer = layers[layer_idx]
            residual = residuals.get((layer_idx, _bid(batch)))
            hidden, residual = layer.neo_preproj(emb, residual)
            residuals[(layer_idx, _bid(batch))] = residual
            return hidden, hidden, hidden

        def attention(q, k, v, batch, cur_layer_id):
            del k, v
            layer = layers[cur_layer_id]
            ubid = _bid_to_ubid[id(batch)]
            pos = positions_list[ubid]
            if per_subbatch_contexts is not None:
                with override_forward_context(per_subbatch_contexts[ubid]):
                    attn_out = layer.neo_attention(
                        positions=pos, hidden_states=q
                    )
            else:
                attn_out = layer.neo_attention(
                    positions=pos, hidden_states=q
                )
            # IDE_006 Step 3.1 — cdec dispatch fork (TSK_015 4.5 / TSK_018 3.1)
            cdec_start, cdec_end = batch.cdec_token_slice
            if cdec_end > cdec_start and cur_layer_id == 0:
                if not getattr(self, "_neo_cdec_fork_logged", False):
                    logger.info(
                        "NEO cdec fork enter: %d cdec rows in sub-batch "
                        "(rows [%d, %d)). Phase 3.2 will route via neo_pacpu.",
                        cdec_end - cdec_start, cdec_start, cdec_end,
                    )
                    self._neo_cdec_fork_logged = True
            return attn_out

        def postproj(attn_out, batch, layer_idx):
            layer = layers[layer_idx]
            residual = residuals[(layer_idx, _bid(batch))]
            hidden, residual = layer.neo_postproj(attn_out, residual)
            residuals[(layer_idx + 1, _bid(batch))] = residual
            return hidden

        callbacks = LayerPipelineCallbacks(
            preproj=preproj, attention=attention, postproj=postproj,
        )
        executor = SubBatchPipelineExecutor(
            num_layers=num_layers, callbacks=callbacks,
        )

        if len(sub_batches) == 1:
            out = executor.forward_sequential(sub_batches[0], embeddings_list[0])
            outputs = [out]
        else:
            out0, out1 = executor.forward_pipeline(sub_batches, embeddings_list)
            outputs = [out0, out1]

        if get_pp_group().is_last_rank:
            finals = []
            for hidden, sb in zip(outputs, sub_batches):
                final_residual = residuals.get((num_layers, _bid(sb)))
                if final_residual is None:
                    final_residual = residuals.get((num_layers - 1, _bid(sb)))
                hidden, _ = self.norm(hidden, final_residual)
                finals.append(hidden)
            return finals
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class GemmaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        # currently all existing Gemma models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings

        self.quant_config = quant_config
        self.model = GemmaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
