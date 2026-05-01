# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_016 Step 5.6 — model expansion sanity checks.

Verifies that each model class exposes the four NEO sub-batch hooks
(``neo_preproj`` / ``neo_attention`` / ``neo_postproj`` on the
DecoderLayer plus ``forward_neo_pipelined`` on the Model).

This is a lightweight class-level introspection test — no weight load,
no forward pass. The runtime correctness of the pipelined path is
covered by the dev e2e smoke (``eval/run_neo_e2e_smoke.py``).
"""

from __future__ import annotations


def _check_decoder_neo_hooks(layer_cls):
    for name in ("neo_preproj", "neo_attention", "neo_postproj"):
        assert callable(getattr(layer_cls, name, None)), (
            f"{layer_cls.__name__} missing required NEO hook: {name}"
        )


def _check_model_neo_forward(model_cls):
    assert callable(getattr(model_cls, "forward_neo_pipelined", None)), (
        f"{model_cls.__name__} missing forward_neo_pipelined"
    )


def test_llama_neo_hooks_present():
    from vllm.model_executor.models.llama import (
        LlamaDecoderLayer, LlamaModel,
    )
    _check_decoder_neo_hooks(LlamaDecoderLayer)
    _check_model_neo_forward(LlamaModel)


def test_qwen2_neo_hooks_present():
    from vllm.model_executor.models.qwen2 import (
        Qwen2DecoderLayer, Qwen2Model,
    )
    _check_decoder_neo_hooks(Qwen2DecoderLayer)
    _check_model_neo_forward(Qwen2Model)


# --- TSK_016 Step 5.6 expansion ----------------------------------------

def test_mistral_inherits_neo_hooks_from_llama():
    """MistralDecoderLayer / MistralModel extend their Llama counterparts —
    NEO hooks come for free via Python MRO."""
    from vllm.model_executor.models.mistral import (
        MistralDecoderLayer, MistralModel,
    )
    _check_decoder_neo_hooks(MistralDecoderLayer)
    _check_model_neo_forward(MistralModel)


def test_phi3_inherits_neo_hooks_from_llama():
    """Phi3ForCausalLM extends LlamaForCausalLM — full inheritance."""
    from vllm.model_executor.models.phi3 import Phi3ForCausalLM
    from vllm.model_executor.models.llama import LlamaForCausalLM
    assert issubclass(Phi3ForCausalLM, LlamaForCausalLM)


def test_gemma_neo_hooks_present():
    """Gemma adds the four NEO hooks directly (own DecoderLayer / Model
    classes — no Llama inheritance)."""
    from vllm.model_executor.models.gemma import (
        GemmaDecoderLayer, GemmaModel,
    )
    _check_decoder_neo_hooks(GemmaDecoderLayer)
    _check_model_neo_forward(GemmaModel)


def test_gemma2_neo_hooks_present():
    """Gemma2 has extra non-residual norms (post_attention_layernorm,
    post_feedforward_layernorm) — verify the hooks are still wired."""
    from vllm.model_executor.models.gemma2 import (
        Gemma2DecoderLayer, Gemma2Model,
    )
    _check_decoder_neo_hooks(Gemma2DecoderLayer)
    _check_model_neo_forward(Gemma2Model)
