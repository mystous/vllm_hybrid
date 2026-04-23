# vLLM Model Execution Flows

This document contains Mermaid sequence diagrams illustrating the execution flow for Llama (Dense) and Mixtral (MoE) models in vLLM.

## 1. Llama (Dense Model) Execution Flow

```mermaid
sequenceDiagram
    participant Runner as GPUModelRunner
    participant Llama as LlamaForCausalLM
    participant Model as LlamaModel
    participant Layer as LlamaDecoderLayer
    participant Attn as LlamaAttention
    participant MLP as LlamaMLP
    participant Kernels as CUDA Kernels (C++)
    participant Logits as LogitsProcessor
    participant Sampler as Sampler

    Runner->>Llama: forward(input_ids, positions, ...)
    Llama->>Model: forward(...)
    
    Note over Model: Embedding Layer
    
    loop For each Decoder Layer
        Model->>Layer: forward(positions, hidden_states)
        
        %% Self Attention Block
        Layer->>Attn: forward(positions, hidden_states)
        Attn->>Kernels: ops.rotary_embedding
        Attn->>Kernels: ops.paged_attention_v1/v2
        Attn-->>Layer: attn_output
        
        %% Feed Forward Block (MLP)
        Layer->>MLP: forward(hidden_states)
        MLP->>Kernels: ops.silu_and_mul
        MLP-->>Layer: mlp_output
    end
    
    Model->>Kernels: ops.rms_norm (Final Norm)
    Model-->>Llama: hidden_states
    Llama-->>Runner: hidden_states

    Runner->>Llama: compute_logits(hidden_states, ...)
    Llama->>Logits: __call__
    Logits-->>Llama: logits
    Llama-->>Runner: logits

    Runner->>Sampler: forward(logits, ...)
    Sampler->>Kernels: flashinfer/sampling kernels
    Sampler-->>Runner: output_tokens
```

## 2. Mixtral (MoE Model) Execution Flow

```mermaid
sequenceDiagram
    participant Runner as GPUModelRunner
    participant Mixtral as MixtralForCausalLM
    participant Model as MixtralModel
    participant Layer as MixtralDecoderLayer
    participant Attn as MixtralAttention
    participant MoE as MixtralMoE
    participant Fused as FusedMoE
    participant Kernels as CUDA Kernels (C++)
    participant Logits as LogitsProcessor
    participant Sampler as Sampler

    Runner->>Mixtral: forward(input_ids, positions, ...)
    Mixtral->>Model: forward(...)
    
    Note over Model: Embedding Layer
    
    loop For each Decoder Layer
        Model->>Layer: forward(positions, hidden_states)
        
        %% Self Attention Block (Same as Llama)
        Layer->>Attn: forward(positions, hidden_states)
        Attn->>Kernels: ops.rotary_embedding
        Attn->>Kernels: ops.paged_attention_v1/v2
        Attn-->>Layer: attn_output
        
        %% MoE Block (Differences Start Here)
        Layer->>MoE: forward(hidden_states)
        MoE->>MoE: gate(hidden_states) -> router_logits
        
        MoE->>Fused: experts(hidden_states, router_logits)
        Note right of Fused: Top-K Gating & Routing
        Fused->>Kernels: ops._moe_C.fused_moe
        Kernels-->>Fused: expert_output
        Fused-->>MoE: final_hidden_states
        MoE-->>Layer: output
    end
    
    Model->>Kernels: ops.rms_norm (Final Norm)
    Model-->>Mixtral: hidden_states
    Mixtral-->>Runner: hidden_states

    Runner->>Mixtral: compute_logits(hidden_states, ...)
    Mixtral->>Logits: __call__
    Logits-->>Mixtral: logits
    Mixtral-->>Runner: logits

    Runner->>Sampler: forward(logits, ...)
    Sampler->>Kernels: flashinfer/sampling kernels
    Sampler-->>Runner: output_tokens
```
