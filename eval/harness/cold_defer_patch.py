#!/usr/bin/env python3
"""IDE_032 배치-인지 deferral: KT_COLD_DEFER=1 이면 select_deferred_experts 가
 - cold(CPU) expert 중 가중치 < KT_COLD_TAU(기본 1.0=전량) 인 것만 deferred,
 - hot(GPU) expert 와 가중치 ≥ τ 인 cold expert 는 immediate.
 (기본 KT 동작 = 가중치 하위 N 개, 장치 무관.) 기존 코드 경로는 env 미설정 시 그대로."""
P="/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"
s=open(P).read()
anchor="""        protected_k = max(0, min(int(protected_k), topk))
        if protected_k == 0:"""
new="""        import os as _os_cd
        if _os_cd.environ.get("KT_COLD_DEFER"):
            # 장치 기준 deferral: cold expert 만 (가중치 < τ) 다음 층으로. hot 은 항상 immediate.
            tau = float(_os_cd.environ.get("KT_COLD_TAU", "1.0"))
            gmask = getattr(self, "gpu_experts_mask", None)
            if gmask is not None:
                gmask = gmask.to(device)
                is_hot = gmask[expert_ids.reshape(-1)].view(batch, topk)
                defer = (~is_hot) & (expert_scores < tau)
                immediate_ids = expert_ids.clone().masked_fill(defer, -1)
                deferred_ids = expert_ids.clone().masked_fill(~defer, -1)
                return immediate_ids, deferred_ids
        protected_k = max(0, min(int(protected_k), topk))
        if protected_k == 0:"""
assert s.count(anchor)==1, s.count(anchor)
s=s.replace(anchor,new); open(P,"w").write(s); print("cold-defer patch applied (env-gated)")
