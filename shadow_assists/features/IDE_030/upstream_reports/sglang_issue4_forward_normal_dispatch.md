# [SGLang] qwen3_moe forward_normal drops expert_location_dispatch_info — silent routing corruption with non-trivial expert placement

**Version**: SGLang 0.5.18
**Symptom**: With `--init-expert-location <map>` (any non-trivial physical_to_logical_map) and `--ep-dispatch-algorithm dynamic`, Qwen3-MoE outputs are garbage. Root cause: `Qwen3MoeSparseMoeBlock.forward_normal` calls `self.topk(hidden_states, router_logits)` without `expert_location_dispatch_info`, so topk ids stay logical while weights are placed physically. The other two paths (forward with forward_batch, op_select_experts) pass it correctly.

**Fix (1-line)**: pass `expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)` in forward_normal, same as the sibling paths.
