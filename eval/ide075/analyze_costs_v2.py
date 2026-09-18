#!/usr/bin/env python3
"""IDE_075 A05/A06 — v2 세션의 FIFO 분해·casebook·producer 층별 비용표·expert 규모/분기 표.
입력: <session>/v2/producer_consumer_metrics_v2.csv.gz (dependency_v2.py --v2), kt_evt.csv(.tasks)
산출: <session>/v2/{queue_wait_breakdown.csv.gz, fifo_dependency_edges.csv.gz, producer_layer_costs.csv, expert_cost_samples.csv.gz, fifo_casebook.md}
규칙: 같은 사건의 timestamp 만 사용. 선행 서비스 span 과 NUMA child span 을 이중 합산하지 않음. 표본 percentile 을 모집단으로 확대하지 않음."""
import gzip, json, os, sys, csv, collections, statistics as st
HOME = os.path.expanduser("~")


def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None


def main(sd):
    v2 = f"{sd}/v2"; rows = list(csv.DictReader(gzip.open(f"{v2}/producer_consumer_metrics_v2.csv.gz", "rt")))
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]; KT = os.environ.get("IDE_KT_HOST", f"{HOME}/.cache/huggingface/kt/ide075")   # IDE_076: 캠페인별 override
    tasks = {int(r["seq"]): r for r in csv.DictReader(l for l in open(f"{KT}/{boot}/kt_evt.csv.tasks") if not l.startswith("#")) if r.get("t_exec_start")}
    def fl(r, k): v = r.get(k); return float(v) if v not in (None, "", "None") else None
    # ---- queue wait breakdown (cold_present_nonempty, v2 필드 존재) ----
    QB = []; edges = []
    for r in rows:
        if r["path_cohort"] != "cold_present_nonempty" or not r.get("producer_def_task_seq"): continue
        seq = int(float(r["producer_def_task_seq"])); tk = tasks.get(seq)
        if not tk: continue
        enq_a = int(tk["t_enq_a"]) / 1e3; ex_s = int(tk["t_exec_start"]) / 1e3; wait = ex_s - enq_a
        pred = []; q = seq - 1
        while q in tasks and q > seq - 64:
            t = tasks[q]; te = int(t["t_exec_end"]) / 1e3; ts_ = int(t["t_exec_start"]) / 1e3
            if te <= enq_a: break
            pred.append((q, t["kind"], max(enq_a, ts_), min(ex_s, te))); q -= 1
        by_kind = collections.Counter()
        for _, kind, a, b in pred: by_kind[{"1": "imm", "2": "done_setter", "3": "deferred"}.get(kind, "other")] += max(0.0, b - a)
        pu = sum(by_kind.values())
        QB.append({"graph_id": r["graph_id"], "replay": r["replay"], "producer_layer": r["producer_layer"], "consumer_layer": r["consumer_layer"], "def_task_seq": seq, "pending_at_enq": tk["pending_at_enq"], "running_seq_at_enq": tk["running_seq_at_enq"], "queue_wait_wall_us": wait, "predecessor_exec_union_us": pu, "pred_deferred_us": by_kind["deferred"], "pred_done_setter_us": by_kind["done_setter"], "pred_imm_us": by_kind["imm"], "pred_other_us": by_kind["other"], "queue_gap_unattributed_us": wait - pu, "n_predecessors_in_window": len(pred), "dequeue_to_exec_us": (int(tk["t_exec_start"]) - int(tk["t_dequeue"])) / 1e3, "enq_bracket_us": (int(tk["t_enq_a"]) - int(tk["t_enq_b"])) / 1e3})
        for q_, kind, a, b in pred: edges.append({"consumer_task_seq": seq, "predecessor_task_seq": q_, "predecessor_kind": kind, "overlap_us": max(0.0, b - a), "producer_layer": r["producer_layer"], "replay": r["replay"], "graph_id": r["graph_id"]})
    for fn, data in (("queue_wait_breakdown.csv.gz", QB), ("fifo_dependency_edges.csv.gz", edges)):
        with gzip.open(f"{v2}/{fn}", "wt", newline="") as f:
            if data: w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    # ---- producer layer costs (graph 별 층별) ----
    PL = []
    for (g, L), rs in sorted(collections.defaultdict(list, {k: [r for r in rows if (r["graph_id"], r["producer_layer"]) == k] for k in set((r["graph_id"], r["producer_layer"]) for r in rows if r["producer_layer"])}).items()):
        rs = [r for r in rs if r["path_cohort"] == "cold_present_nonempty"]
        if not rs: continue
        def s(k, q=.5): v = [fl(r, k) for r in rs if fl(r, k) is not None]; return pct(v, q)
        PL.append({"graph_id": g, "step": rs[0]["step_name"], "producer_layer": L, "consumer_layer": int(L) + 1, "n": len(rs), "n_cold_assignments_p50": s("n_cold_assignments"), "unique_cold_experts_numa0_p50": s("n_unique_cold_experts_numa0"), "unique_cold_experts_numa1_p50": s("n_unique_cold_experts_numa1"), "max_rows_numa0_p50": s("max_rows_numa0"), "sum_rows_numa0_p50": s("sum_rows_numa0"), "sum_rows_numa1_p50": s("sum_rows_numa1"),
                   "n_amx_numa0_p50": s("n_amx_numa0"), "n_avx_numa0_p50": s("n_avx_numa0"), "go_to_enqueue_p50": s("go_to_enqueue_us"), "enqueue_to_start_p50": s("enqueue_to_start_us"), "predecessor_exec_union_p50": s("predecessor_exec_union_us"), "queue_gap_unattributed_p50": s("queue_gap_unattributed_us"), "deferred_service_span_p50": s("deferred_service_span_us"), "deferred_service_span_p95": s("deferred_service_span_us", .95),
                   "numa0_span_p50": s("numa0_span_us"), "numa1_span_p50": s("numa1_span_us"), "numa_skew_p50": s("numa_completion_skew_us"), "producer_end_to_pub_p50": s("producer_end_to_pub_us"), "cold_pub_delta_p50": s("cold_pub_delta_us"), "cold_pub_delta_p95": s("cold_pub_delta_us", .95), "gpu_pre_h2d_gap_p50": s("gpu_pre_h2d_gap_us"), "gpu_idle_inside_gap_p50": s("gpu_idle_inside_gap_us"), "h2d_duration_p50": s("h2d_duration_us"), "overlap_budget_p50": s("overlap_budget_us"),
                   "late_share": sum(1 for r in rs if (fl(r, "cold_pub_delta_us") or 0) > 0) / len(rs), **{f"stage_numa0_{k_}_p50": s(f"stage_numa0_{k_}") for k_ in ("prepare", "cpy_input", "q_input", "up_gate", "act", "q_down", "down", "weight", "total")}, "observer_mode": m["mode"], "sample_probability": 1.0, "validity": "OK(전수)"})
    with open(f"{v2}/producer_layer_costs.csv", "w", newline="") as f:
        if PL: w = csv.DictWriter(f, fieldnames=list(PL[0].keys())); w.writeheader(); w.writerows(PL)
    # ---- expert cost samples: 작업 단위 규모·분기·단계 (전수; expert 별 rows 는 미수집 → 작업 단위) ----
    EX = [{"graph_id": r["graph_id"], "replay": r["replay"], "producer_layer": r["producer_layer"], "def_task_seq": r.get("producer_def_task_seq"), "n_cold_assignments": r.get("n_cold_assignments"), "unique_numa0": r.get("n_unique_cold_experts_numa0"), "unique_numa1": r.get("n_unique_cold_experts_numa1"), "max_rows_numa0": r.get("max_rows_numa0"), "max_rows_numa1": r.get("max_rows_numa1"), "sum_rows_numa0": r.get("sum_rows_numa0"), "sum_rows_numa1": r.get("sum_rows_numa1"), "n_amx_numa0": r.get("n_amx_numa0"), "n_avx_numa0": r.get("n_avx_numa0"), "n_amx_numa1": r.get("n_amx_numa1"), "n_avx_numa1": r.get("n_avx_numa1"), "kernel_branch_note": "n_amx/n_avx = do_gate_up_gemm(2)+do_down_gemm(1) per expert, ith==0 계수", "deferred_service_span_us": r.get("deferred_service_span_us"), "numa0_span_us": r.get("numa0_span_us"), "numa1_span_us": r.get("numa1_span_us"), **{f"stage_numa0_{k_}": r.get(f"stage_numa0_{k_}") for k_ in ("prepare", "cpy_input", "q_input", "up_gate", "act", "q_down", "down", "weight", "total")}, **{f"stage_numa1_{k_}": r.get(f"stage_numa1_{k_}") for k_ in ("prepare", "cpy_input", "q_input", "up_gate", "act", "q_down", "down", "weight", "total")}, "inclusive_or_exclusive": "stage 는 순차 구간(prefill 경로), total 은 inclusive", "validity": "OK" if r["path_cohort"] == "cold_present_nonempty" else r["path_cohort"]} for r in rows if r.get("producer_layer")]
    with gzip.open(f"{v2}/expert_cost_samples.csv.gz", "wt", newline="") as f:
        if EX: w = csv.DictWriter(f, fieldnames=list(EX[0].keys())); w.writeheader(); w.writerows(EX)
    # ---- casebook ----
    big = [r for r in rows if r["path_cohort"] == "cold_present_nonempty" and r.get("cold_pub_delta_us") and r["step_name"].startswith("step[DECODE bs=6")]
    small = [r for r in rows if r["path_cohort"] == "cold_present_nonempty" and r.get("cold_pub_delta_us") and not r["step_name"].startswith("step[DECODE bs=6")]
    empty = [r for r in rows if r["path_cohort"] == "cold_path_empty"]
    def pick(rs, q):
        if not rs: return None
        v = sorted(rs, key=lambda r: float(r["cold_pub_delta_us"])); return v[min(len(v) - 1, int(q * (len(v) - 1)))]
    cases = [("큰 배치 전형 (p50)", pick(big, .5)), ("큰 배치 p95 부근", pick(big, .95)), ("큰 배치 최대 tail", pick(big, 1.0)), ("작은 배치 (숨겨진 계산)", pick(small, .5)), ("empty-cold", empty[0] if empty else None)]
    L = [f"# fifo_casebook — {os.path.basename(sd)} ({m['mode']})", "", "각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).", ""]
    for title, r in cases:
        if not r: L += [f"## {title}", "", "해당 표본 없음", ""]; continue
        seq = int(float(r["producer_def_task_seq"])) if r.get("producer_def_task_seq") else None; qb = next((q for q in QB if q["def_task_seq"] == seq), None)
        L += [f"## {title}", "", f"- ID: graph {r['graph_id']} replay {r['replay']} consumer layer {r['consumer_layer']} ← producer layer {r['producer_layer']} slot {r['slot']} epoch {r['epoch']} def_task_seq {seq} step {r['step_name']}",
              f"- 부분순서 (µs, 절대): go(p) {r.get('t_go_us')} → enqueue+{r.get('go_to_enqueue_us')} → exec_start +{r.get('enqueue_to_start_us')} (선행 실행 점유 {r.get('predecessor_exec_union_us')}, 미분리 {r.get('queue_gap_unattributed_us')}) → service {r.get('deferred_service_span_us')} (numa0 {r.get('numa0_span_us')} / numa1 {r.get('numa1_span_us')}, skew {r.get('numa_completion_skew_us')}) → pub +{r.get('producer_end_to_pub_us')}",
              f"- GPU: hot_ready {r['t_hot_ready_us']} / pub {r.get('t_pub_before_us')} (delta {r.get('cold_pub_delta_us')}) / h2d_start {r['t_h2d_start_us']} (pre gap {r['gpu_pre_h2d_gap_us']}, idle {r['gpu_idle_inside_gap_us']}) / h2d {r['h2d_duration_us']} / combine +{r.get('gpu_post_h2d_gap_us')}",
              f"- 규모: cold assignments {r.get('n_cold_assignments')}, unique numa0/1 {r.get('n_unique_cold_experts_numa0')}/{r.get('n_unique_cold_experts_numa1')}, max rows {r.get('max_rows_numa0')}/{r.get('max_rows_numa1')}, AMX/AVX numa0 {r.get('n_amx_numa0')}/{r.get('n_avx_numa0')}; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = " + ",".join(str(r.get(f'stage_numa0_{k_}')) for k_ in ("prepare", "cpy_input", "q_input", "up_gate", "act", "q_down", "down", "weight", "total")),
              f"- 선행 task 분해: {json.dumps({k: qb[k] for k in ('pred_deferred_us', 'pred_done_setter_us', 'pred_imm_us', 'pred_other_us', 'n_predecessors_in_window', 'pending_at_enq')}) if qb else 'NOT_LINKED'}", f"- clock: {r['clock_validity']}; 매핑: {r['mapping_method']}", ""]
    open(f"{v2}/fifo_casebook.md", "w").write("\n".join(L) + "\n")
    print("QB", len(QB), "edges", len(edges), "layers", len(PL), "expert rows", len(EX))
    if QB:
        for k in ("queue_wait_wall_us", "predecessor_exec_union_us", "pred_deferred_us", "pred_done_setter_us", "queue_gap_unattributed_us", "dequeue_to_exec_us", "enq_bracket_us"):
            v = [q[k] for q in QB]; print(f"  {k}: p50 {pct(v, .5):.1f} p95 {pct(v, .95):.1f} max {max(v):.1f}")


if __name__ == "__main__": main(sys.argv[1])
