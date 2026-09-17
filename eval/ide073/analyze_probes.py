#!/usr/bin/env python3
"""IDE_073 — 프로브 산출 분석 (§7.4). 입력: <model>-PROBES/{bootA/D0,D2,D3; bootB/D1} + torch profiler 트레이스 (Chrome trace json.gz).
출력 profiles/<model>/: probe_map.md, gpu_activities.csv, layer_step_raw.csv(decode step 별 GPU 활동 요약), layer_stage_aggregate.md, cpu_jobs.csv(복사), critical_path_intervals.csv(PARTIAL), observer_overhead.csv.
가정을 명시: 트레이스의 kernel 이름으로 stage 를 태깅(추정 아님, 이름 그대로 기록); 층 경계는 GPU expert(fused_moe) 커널 등장 순서로 구분; 의존 완료 시각은 cuStreamWaitValue/memop 을 트레이스에서 직접 볼 수 없으면 PARTIAL_DEPENDENCY_MEASUREMENT."""
import gzip, json, os, sys, glob, csv, statistics as st, re, collections
FEAT = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_073"); CAMP = os.path.expanduser("~/projects/vllm_hybrid/eval/results/IDE_073_20260917"); PROF = os.path.expanduser("~/.cache/huggingface/kt/ide073/profiles")

def analyze_trace(model, tf, out):
    d = json.load(gzip.open(tf, "rb")); ev = d.get("traceEvents", d if isinstance(d, list) else [])
    rows = []; kern = []
    for e in ev:
        if e.get("ph") != "X": continue
        cat = e.get("cat", ""); nm = e.get("name", ""); a = e.get("args", {})
        rows.append([cat, nm[:120], e.get("ts"), e.get("dur"), a.get("stream"), a.get("correlation"), e.get("pid"), e.get("tid")])
        if cat in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"): kern.append((e["ts"], e["ts"] + e.get("dur", 0), cat, nm, a.get("stream")))
    prof_id = os.path.basename(tf).split('.')[0][-24:].split("-TP")[0]
    for old in glob.glob(f"{out}/gpu_activities_*.csv*"):   # 이전 profile id 의 산출 제거 (같은 id 의 다른 rank 는 유지)
        if prof_id not in os.path.basename(old): os.remove(old)
    with gzip.open(f"{out}/gpu_activities_{os.path.basename(tf).split('.')[0][-24:]}.csv.gz", "wt") as f:
        w = csv.writer(f); w.writerow(["cat", "name", "ts_us", "dur_us", "stream", "correlation", "pid", "tid"]); w.writerows(rows)
    kern.sort()
    ann = [k for k in kern if k[2] == "gpu_user_annotation"]; kern = [k for k in kern if k[2] != "gpu_user_annotation"]   # busy·gap 은 실제 GPU 작업(kernel/memcpy/memset)만; annotation 은 이름별 합계 표에만
    # 커널 이름별 합계 (GPU 시간 분해; 이름은 트레이스 원문)
    agg = collections.defaultdict(lambda: [0, 0.0])
    for s0, s1, cat, nm, stream in kern: agg[(cat, nm[:90])][0] += 1; agg[(cat, nm[:90])][1] += (s1 - s0)
    total_gpu = sum(v[1] for v in agg.values())
    for s0, s1, cat, nm, stream in ann: agg[(cat, nm[:90])][0] += 1; agg[(cat, nm[:90])][1] += (s1 - s0)
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:60]
    # 커널 간 GPU 유휴 간격 (같은 rank 트레이스 내 모든 스트림 union 의 빈 구간)
    gaps = []
    if kern:
        cur_end = kern[0][1]
        for s0, s1, cat, nm, stream in kern[1:]:
            if s0 > cur_end: gaps.append(s0 - cur_end)
            cur_end = max(cur_end, s1)
    span = (kern[-1][1] - kern[0][0]) if kern else 0
    # 층 경계 추정: MoE GPU expert 커널(fused_moe/grouped gemm 이름) 등장으로 층 구분 → 층 사이 GPU 빈 간격 = 결합 전 대기 후보 (PARTIAL)
    moe_idx = [i for i, k in enumerate(kern) if re.search(r"fused_moe|moe_align|grouped|group_gemm|silu_and_mul|moe_sum", k[3], re.I)]
    layer_gaps = []
    for i in range(1, len(moe_idx)):
        a = kern[moe_idx[i - 1]]; b = kern[moe_idx[i]]
        # a 종료 후 b 시작 전 구간의 GPU 유휴 합
        idle = 0; cur = a[1]
        for k in kern[moe_idx[i - 1] + 1: moe_idx[i]]:
            if k[0] > cur: idle += k[0] - cur
            cur = max(cur, k[1])
        layer_gaps.append(idle)
    def pct(v, q): v = sorted(v); return v[int(q * (len(v) - 1))] if v else None
    return {"trace": os.path.basename(tf), "events": len(rows), "gpu_ops": len(kern), "gpu_busy_us": total_gpu, "span_us": span, "gpu_busy_fraction": (total_gpu / span) if span else None,
            "gap_count": len(gaps), "gap_p50_us": pct(gaps, .5), "gap_p95_us": pct(gaps, .95), "gap_p99_us": pct(gaps, .99), "gap_max_us": max(gaps) if gaps else None, "gap_sum_us": sum(gaps),
            "moe_kernel_count": len(moe_idx), "inter_moe_idle_p50_us": pct(layer_gaps, .5), "inter_moe_idle_p95_us": pct(layer_gaps, .95), "inter_moe_idle_p99_us": pct(layer_gaps, .99), "inter_moe_idle_max_us": max(layer_gaps) if layer_gaps else None,
            "top_kernels": [(cat, nm, n, round(t, 1)) for (cat, nm), (n, t) in top]}

def main():
    model = sys.argv[1]; out = f"{FEAT}/profiles/{model}"; os.makedirs(out, exist_ok=True)
    base = f"{CAMP}/{model}-PROBES"; ps = json.load(open(f"{base}/probes_summary.json")) if os.path.exists(f"{base}/probes_summary.json") else {"sessions": {}}
    L = [f"# layer/stage aggregate — {model} (프로브 산출의 기계적 집계)", ""]
    # observer overhead
    with open(f"{out}/observer_overhead.csv", "w") as f:
        w = csv.writer(f); w.writerow(["session", "output_tps", "duration_s", "throughput_ratio", "elapsed_ratio", "completed", "failed"])
        for k, v in ps["sessions"].items(): sm = v.get("summary") or {}; w.writerow([k, sm.get("output_throughput"), sm.get("duration"), v.get("throughput_ratio"), v.get("elapsed_ratio"), sm.get("completed"), sm.get("failed")])
    # D1 cpu jobs
    cj = f"{base}/bootB/D1/cpu_jobs.csv"
    if os.path.exists(cj):
        import shutil; shutil.copy(cj, f"{out}/cpu_jobs.csv"); rows = list(csv.DictReader(open(cj)))
        L += ["## CPU expert 작업 (D1, KT_PHASE_PROF 64회당 1회 표본, µs; numa 서브풀별 1행 = 층 작업의 절반)", "", f"표본 {len(rows)} 행", "", "| qlen 구간 | n | activated_expert p50 | total p50/p95/p99 | up_gate p50 | down p50 | q_input p50 | cpy_input p50 | weight p50 |", "|---|---|---|---|---|---|---|---|---|"]
        def pct(v, q): v = sorted(v); return v[int(q * (len(v) - 1))] if v else None
        for lo, hi, lab in ((0, 80, "decode(qlen≤80)"), (80, 10 ** 9, "prefill(qlen>80)")):
            rs = [r for r in rows if lo < int(r["qlen"]) <= hi]
            if not rs: continue
            g = lambda k: [int(r[k]) for r in rs]
            L.append(f"| {lab} | {len(rs)} | {pct(g('activated_expert'), .5)} | {pct(g('total_us'), .5)}/{pct(g('total_us'), .95)}/{pct(g('total_us'), .99)} | {pct(g('up_gate_us'), .5)} | {pct(g('down_us'), .5)} | {pct(g('q_input_us'), .5)} | {pct(g('cpy_input_us'), .5)} | {pct(g('weight_us'), .5)} |")
        # activated expert 별 total 중앙값 (decode)
        by = collections.defaultdict(list)
        for r in rows:
            if int(r["qlen"]) <= 80: by[int(r["activated_expert"])].append(int(r["total_us"]))
        L += ["", "activated_expert 별 total µs 중앙값 (decode): " + ", ".join(f"{k}:{st.median(v):.0f}(n={len(v)})" for k, v in sorted(by.items())), ""]
        tq = [l for l in open(f"{base}/bootB/D1/cpu_jobs_raw.txt") if "[kt-tq]" in l] if os.path.exists(f"{base}/bootB/D1/cpu_jobs_raw.txt") else []
        if tq: L += ["[kt-tq] TaskQueue 집계 (마지막 3행):", "", "```"] + [l.strip()[:300] for l in tq[-3:]] + ["```", ""]
    else: L += ["## CPU expert 작업: NOT_COLLECTED (D1 미실행 또는 표본 없음)", ""]
    # D2 traces
    traces = sorted(glob.glob(f"{PROF}/{model}/*.trace.json.gz"))
    for bd in ("bootA_retry2", "bootA_retry", "bootA"):   # 가장 최근 D2 세션이 남긴 트레이스만 사용
        mp = f"{base}/{bd}/D2/metrics.json"
        if os.path.exists(mp) and json.load(open(mp)).get("trace_files"):
            names = {t[0] for t in json.load(open(mp))["trace_files"]}; traces = [t for t in traces if os.path.basename(t) in names]; break
    if traces:
        L += ["## GPU 타임라인 (D2, SGLang torch profiler CPU+CUDA activity; rank 별 트레이스)", "", "| rank trace | events | GPU ops | GPU busy s | span s | busy 비율 | 커널 간 유휴 gap: n / p50 / p95 / p99 / max µs / 합 s | MoE 커널 수 | MoE 커널 사이 유휴 p50/p95/p99/max µs |", "|" + "---|" * 9]
        allres = []
        for tf in traces[:8]:
            try: r = analyze_trace(model, tf, out); allres.append(r)
            except Exception as e: L.append(f"| {os.path.basename(tf)} | 파싱 실패 {str(e)[:80]} |"); continue
            L.append(f"| {r['trace'][:40]} | {r['events']} | {r['gpu_ops']} | {r['gpu_busy_us']/1e6:.3f} | {r['span_us']/1e6:.3f} | {r['gpu_busy_fraction'] and round(r['gpu_busy_fraction'],3)} | {r['gap_count']} / {r['gap_p50_us']} / {r['gap_p95_us']} / {r['gap_p99_us']} / {r['gap_max_us']} / {r['gap_sum_us']/1e6:.3f} | {r['moe_kernel_count']} | {r['inter_moe_idle_p50_us']}/{r['inter_moe_idle_p95_us']}/{r['inter_moe_idle_p99_us']}/{r['inter_moe_idle_max_us']} |")
        if allres:
            L += ["", "### 상위 GPU 커널 (첫 rank 트레이스, 이름 원문, 호출 수, 합계 µs)", "", "| cat | kernel | n | sum µs |", "|---|---|---|---|"] + [f"| {c} | `{n[:80]}` | {k} | {t} |" for c, n, k, t in allres[0]["top_kernels"][:40]]
        L += ["", "의존 대기(cold_exposed_wait 등): 트레이스에 CPU→GPU done 신호(memop)·H2D 복사 완료 이벤트가 커널과 별도 항목으로 식별되지 않아 `PARTIAL_DEPENDENCY_MEASUREMENT` — 위 'MoE 커널 사이 GPU 유휴' 가 결합 전 대기의 상한 관측치이며, hot/cold 어느 경로가 늦었는지는 이 트레이스만으로 분리되지 않음.", ""]
        with open(f"{out}/critical_path_intervals.csv", "w") as f:
            w = csv.writer(f); w.writerow(["trace", "metric", "value_us", "status"]);
            for r in allres:
                for k in ("gap_p50_us", "gap_p95_us", "gap_p99_us", "gap_max_us", "inter_moe_idle_p50_us", "inter_moe_idle_p95_us", "inter_moe_idle_p99_us", "inter_moe_idle_max_us"): w.writerow([r["trace"], k, r[k], "PARTIAL_DEPENDENCY_MEASUREMENT"])
    else: L += ["## GPU 타임라인: NOT_COLLECTED (D2 트레이스 없음)", ""]
    # D3
    d3 = f"{base}/bootA_retry/D3" if os.path.exists(f"{base}/bootA_retry/D3/perf_report_top.txt") else f"{base}/bootA/D3"
    if os.path.exists(f"{d3}/perf_report_top.txt"):
        top = [l for l in open(f"{d3}/perf_report_top.txt") if re.match(r"\s+\d+\.\d+%", l)][:25]
        L += ["## CPU 타임라인 (D3, perf record 99 Hz 20 s, 스케줄러 4 프로세스; 상위 25 심볼)", "", "```"] + [l.rstrip()[:160] for l in top] + ["```", ""]
        if os.path.exists(f"{d3}/perf_stat_20s.txt"): L += ["perf stat 20 s:", "", "```"] + [l.rstrip() for l in open(f"{d3}/perf_stat_20s.txt") if l.strip() and not l.startswith("#")] + ["```", ""]
        if os.path.exists(f"{d3}/pcm_memory.csv"):
            rows = list(csv.reader(open(f"{d3}/pcm_memory.csv")))
            if len(rows) > 2:
                h = rows[1]; g = rows[0]; idx = [i for i in range(len(h)) if g[i] == "System" and h[i] in ("Read", "Write")]
                vals = [[float(r[i]) if r[i].strip() not in ("", "N/A") else 0 for i in idx] for r in rows[2:] if len(r) >= len(h)]
                if vals: L += [f"pcm-memory (2 s, MB/s 단위는 pcm-memory 헤더 정의 'System Read/Write'): 샘플 {len(vals)}, 읽기 평균 {st.mean(v[0] for v in vals):.0f} MB/s, 쓰기 평균 {st.mean(v[1] for v in vals):.0f} MB/s, 읽기 최대 {max(v[0] for v in vals):.0f} MB/s", ""]
    else: L += ["## CPU 타임라인: NOT_COLLECTED", ""]
    # probe map
    PM = ["# probe_map — " + model, "", "| 구간 | 실제 경로 (파일:함수) | 계측 수단 | 상태 |", "|---|---|---|---|",
          "| S00 요청 수신·토큰화·스케줄 | sglang tokenizer_manager / scheduler (HTTP 프로세스, sglang::scheduler_TP*) | 요청별 TTFT (client) | 부분 (서버 내부 시각 미수집) |",
          "| S01 attention·MoE 입력 | glm4_moe.py/qwen3_moe.py forward → triton/flashinfer attention 커널 | D2 커널명 | 이름 수준 |",
          "| S02 router·top-k | sglang/srt/layers/moe/topk.py select_experts → fused_moe_triton layer.py:376 KTEPWrapperMethod.apply (kt_ep_wrapper.py:370-430) | D2 커널명 | 이름 수준 |",
          "| S03 CPU 입력 전달 | kt_kernel/experts_base.py submit_forward: pinned slot ring (KExpertsCPUBuffer, slot=layer_idx%depth) .copy_(non_blocking) D2H | D2 gpu_memcpy | bytes 는 트레이스 args |",
          "| S04 CPU 작업 전달·대기 | callback-free: cpuinfer.h go_on_stream (memop) → poll_loop_ (poller 스레드) → TaskQueue enqueue; task_queue.cpp KT_TQ_TIMING (wait_us) | D1 [kt-tq] | 집계만 |",
          "| S05 CPU expert 계산 | operators/amx/moe_base.hpp forward_* → moe.hpp do_gate_up_gemm/do_down_gemm (qlen>MIN_QLEN or m>=MIN_ROWS → amx::mat_mul, 아니면 amx::vec_mul(AVX-512 VNNI)); KT_PHASE_PROF (prepare/cpy_input/q_input/up_gate/act/q_down/down/weight) | D1 cpu_jobs.csv | 64회당 1회 표본, 층 id 없음 |",
          "| S06 CPU 완료·반환 | cpuinfer.h done 플래그 → experts_base.py sync_forward wait_done_on_stream → output_gpu.copy_(output_cpu, non_blocking) H2D | D2 gpu_memcpy | 부분 |",
          "| S07 GPU hot expert | fused_moe_triton (kernel 이름 fused_moe_kernel 등) | D2 | 이름 수준 |",
          "| S08 결합 전 의존 대기 | kt_ep_wrapper.py apply: output = gpu + cpu_output (sync) | D2 MoE 커널 사이 유휴 | PARTIAL_DEPENDENCY_MEASUREMENT |",
          "| S09 결합·TP 통신·잔차 | all-reduce (custom allreduce 커널), residual add | D2 커널명 | 이름 수준 |",
          "| S10 sampling·출력 | sampler 커널, detokenizer 프로세스 | D2 / client | 부분 |",
          "| S11 graph 경로 | decode_cuda_graph_runner replay (capture bs 목록 runtime_proofs) | D2 (graph 내부 커널은 CUPTI 로 기록) | 이름 수준 |",
          "| S12 동적 배치·layerwise prefill | 설치본 미지원 | — | NOT_APPLICABLE |"]
    open(f"{out}/probe_map.md", "w").write("\n".join(PM) + "\n"); open(f"{out}/layer_stage_aggregate.md", "w").write("\n".join(L) + "\n"); print(out, len(L), "lines")

if __name__ == "__main__":
    main()
