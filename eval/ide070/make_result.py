#!/usr/bin/env python3
"""IDE_070 — 전 실험 결과를 평가·의견 없이 표로 수집 → shadow_assists/features/IDE_070/RESULT.md
사용: ~/venv-bench/bin/python eval/ide070/make_result.py
"""
import glob, json, os, re, statistics as st, collections
REPO = os.path.expanduser("~/projects/vllm_hybrid"); R = f"{REPO}/eval/results"
OUT = f"{REPO}/shadow_assists/features/IDE_070/RESULT.md"
L = []
def P(s=""): L.append(s)

def bench(d):
    f = f"{d}/bench_summary.txt"
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None
    t = open(f).read()
    def g(k):
        m = re.search(rf"{k}[^:]*:\s*([\d.]+)", t); return float(m.group(1)) if m else None
    return dict(tput=g("Output token throughput"), tot=g("Total token throughput"), dur=g("Benchmark duration"),
                ttft=g("Median TTFT"), ttft95=g("P95 TTFT"), tpot=g("Median TPOT"), tpot95=g("P95 TPOT"),
                ok=g("Successful requests"), fail=g("Failed requests"))
def cpu(d):
    f = f"{d}/cpu_util.txt"
    if not os.path.exists(f): return None
    v = [float(l.split("busy=")[1]) for l in open(f) if "busy=" in l]
    return st.mean(v) if v else None
def hbm(cell):
    f = f"{cell}/hbm_after_boot.csv"
    if not os.path.exists(f): return "—"
    vals = [l.split(",")[1].strip().split()[0] for l in open(f) if l.strip()][:4]
    return "/".join(vals)
def launch(cell):
    f = f"{cell}/launch_cmd.txt"
    return open(f).read().strip() if os.path.exists(f) else ""
def verdict(cell):
    f = f"{cell}/verdict.txt"
    return open(f).read().strip() if os.path.exists(f) else ("HEALTH_OK" if glob.glob(f"{cell}/rep*/bench_summary.txt") else "?")
def fmt(x, n=1): return "—" if x is None else f"{x:.{n}f}"
def msd(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return "—"
    if len(vals) == 1: return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f} ± {st.pstdev(vals):.1f}"

def cell_rows(base, names=None):
    cells = sorted(d for d in glob.glob(f"{base}/*/") if os.path.isdir(d) and (os.path.exists(d + "verdict.txt") or glob.glob(d + "rep*/bench_summary.txt") or glob.glob(d + "bench_summary.txt")))
    rows = []
    for c in cells:
        name = os.path.basename(c.rstrip("/"))
        reps = sorted(glob.glob(c + "rep*/")) or [c]
        bs = [(os.path.basename(r.rstrip("/")), bench(r), cpu(r)) for r in reps]
        extra = [(os.path.basename(r.rstrip("/")), bench(r), cpu(r)) for r in sorted(glob.glob(c + "c*/")) if re.match(r"c\d+$", os.path.basename(r.rstrip("/")))]
        rows.append((name, verdict(c), bs, extra, hbm(c), launch(c), c))
    return rows

def series_table(title, base, note=""):
    if not base or not os.path.isdir(base): P(f"### {title}\n\n(결과 디렉터리 없음)\n"); return
    P(f"### {title}\n"); P(f"디렉터리 `eval/results/{os.path.basename(base.rstrip('/'))}/`. {note}\n")
    P("| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for name, vd, bs, extra, hb, lc, c in cell_rows(base):
        first = True
        for rn, b, cu in bs + extra:
            if b is None:
                P(f"| {name if first else ''} | {vd if first else ''} | {rn} | (없음) | | | | | | {hb if first else ''} |"); first = False; continue
            P(f"| {name if first else ''} | {vd if first else ''} | {rn} | {fmt(b['tput'],2)} | {fmt(b['tot'],1)} | {fmt(b['tpot'])}/{fmt(b['tpot95'])} | {fmt(b['ttft'],0)}/{fmt(b['ttft95'],0)} | {fmt(b['ok'],0)}/{fmt(b['fail'],0)} | {fmt(cu)} | {hb if first else ''} |")
            first = False
        tp = [b["tput"] for _, b, _ in bs if b]
        if len(tp) > 1: P(f"| | | 평균±표준편차 | **{msd(tp)}** | | {msd([b['tpot'] for _, b, _ in bs if b])} / {msd([b['tpot95'] for _, b, _ in bs if b])} | {msd([b['ttft'] for _, b, _ in bs if b])} / {msd([b['ttft95'] for _, b, _ in bs if b])} | | | |")
        if not bs and not extra: P(f"| {name} | {vd} | — | (벤치 없음) | | | | | | {hb} |")
    P()
    P("<details><summary>launch 명령</summary>\n")
    for name, vd, bs, extra, hb, lc, c in cell_rows(base):
        if lc: P(f"- `{name}`: `{lc}`")
        ef = f"{c}/error_lines.log"
        if vd.split()[0] != "HEALTH_OK" and os.path.exists(ef) and os.path.getsize(ef):
            P(f"  - error_lines: `{open(ef).read().strip()[:400]}`")
    P("\n</details>\n")

def latest(pattern):
    ds = sorted(glob.glob(f"{R}/{pattern}")); return ds[-1] if ds else None
def all_of(pattern): return sorted(glob.glob(f"{R}/{pattern}"))

# ---------------- 문서 ----------------
P("# IDE_070 결과 — CPU offloading 지시서(cpu_offload_no_01) 전 항목 실행 결과\n")
P("측정값만 기록한다. 평가·해석·의견은 포함하지 않는다. 노드 violet-h100-016, 브랜치 feat/cpu-offload-diag, 2026-09-15.\n")
P("## 0. 공통 조건\n")
P("- 모델: Qwen3-Coder-480B-A35B-Instruct-FP8 (GPU: attention·dense·hot expert), CPU expert: kt INT4 (`/models/kt/qwen3-480b-int4`, AMXINT4). E 시리즈만 GPU 측 체크포인트 QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ (4-bit g128).")
P("- 서빙: SGLang 0.5.18 (+로컬 수정 10 파일) + kt-kernel 0.7.0.post1 (컨테이너 `sgl-kt`), TP4 (GPU 0,1,2,3; A6 는 0,1,4,5), H100 80GB ×4, Xeon 8480+ ×2 (turbo OFF 2.0 GHz, BIOS 잠금), DDR5-4400 8ch×2DPC/소켓 (2 TB).")
P("- 기준 구성: hot expert 96 (IDE_069 hotmap), deferred 4, cuda graph bs≤64 (prefill graph 비활성), mem-fraction 0.95, max-total-tokens 40,960, `--ep-dispatch-algorithm dynamic`, cpuinfer 96, threadpool 2.")
P("- 벤치: `vllm bench serve` (컨테이너 vllm-h100), sonnet 512/128 prefix 100, seed 42, C=64 고정 동시성, 256 요청, 셀당 워밍업 1회 (C16 32요청) 후 3회 반복. 반복 간 `/flush_cache` 없음 (같은 seed 프롬프트 재사용). d4 의 C224 셀만 flush 후 896 요청.")
P("- CPU busy = /proc/stat 2 s 샘플 전체 코어 평균. HBM = 부팅 직후 `nvidia-smi` memory.used.")
P("- 하드웨어·소프트웨어 상세 프로브: `shadow_assists/features/IDE_069/hwsw/` (lscpu, dmidecode, nvidia_topo, 컨테이너 패키지 목록).\n")

P("## 1. 측정 (TSK_054)\n")
m1 = latest("*_ide070_measure_baseline")
if m1 and os.path.exists(f"{m1}/MEASURE_SUMMARY.md"):
    P(f"### 1.1 1차 측정 (`eval/results/{os.path.basename(m1)}/MEASURE_SUMMARY.md` 전문)\n")
    txt = open(f"{m1}/MEASURE_SUMMARY.md").read().split("\n", 1)[1]
    txt = txt.replace("## ", "#### ")
    P(txt)
    P("- 주: pcm 표의 DRAM 열은 pcm CSV 의 READ/WRITE (2 s 간격당 GB) 를 2 로 나눈 GB/s. rep3 의 perf stat 은 창 400 s (부하 52 s + 유휴) 이며 `-p` 대상은 HTTP 서버 프로세스 (스케줄러 아님).\n")
m2 = latest("*_ide070_measure2")
if m2:
    P(f"### 1.2 2차 측정 (`eval/results/{os.path.basename(m2)}/`)\n")
    for r in ("rep1_perfstat", "rep2_profile", "rep3_pcmmem"):
        b = bench(f"{m2}/{r}")
        if b: P(f"- {r}: {b['tput']:.2f} tok/s, TPOT p50/p95 {b['tpot']:.1f}/{b['tpot95']:.1f} ms, TTFT p50/p95 {b['ttft']:.0f}/{b['ttft95']:.0f} ms" + (" (py-spy 200 Hz + perf record 동시 실행 중)" if r == "rep2_profile" else ""))
    st_ = f"{m2}/tp0_status.txt"
    if os.path.exists(st_): P("- TP0 (`sglang::scheduler_TP0`, GPU0 compute 프로세스): " + open(st_).read().strip().replace("\n", " · ").replace("\t", " "))
    aff = f"{m2}/tp0_thread_affinity.txt"
    if os.path.exists(aff):
        w0 = sorted(int(l.split()[2]) for l in open(aff) if l.split()[1].startswith("numa_0_t")); w1 = sorted(int(l.split()[2]) for l in open(aff) if l.split()[1].startswith("numa_1_t"))
        P(f"- kt 워커 스레드 고정: numa_0_t ×{len(w0)} → cpu {w0[0]}–{w0[-1]}, numa_1_t ×{len(w1)} → cpu {w1[0]}–{w1[-1]}; 마스터 numa_0_m_0 → cpu 0, numa_1_m_0 → cpu 56.")
    ns = f"{m2}/rep1_perfstat/numastat_tp0_after.txt"
    if os.path.exists(ns):
        for l in open(ns):
            if l.startswith("Private") or l.startswith("Total"): P(f"- numastat TP0 {l.split()[0]}: node0 {l.split()[1]} MB, node1 {l.split()[2]} MB, 합계 {l.split()[3]} MB")
    # perf record 분류
    full = f"{m2}/rep2_profile/perf_report_full.txt"
    if os.path.exists(full):
        cat = collections.Counter(); bycomm = collections.Counter(); tot = 0
        for ln in open(full):
            m = re.match(r"\s*([\d.]+)%\s+(\S+)\s+(\S+)\s+\[(.)\]\s+(.*)", ln)
            if not m: continue
            pct = float(m.group(1)); comm = m.group(2); dso = m.group(3); sym = m.group(5).strip(); tot += pct
            bycomm["numa_*_t (kt 워커)" if re.match(r"numa_\d_t", comm) else ("numa_*_m (kt 마스터)" if re.match(r"numa_\d_m", comm) else comm)] += pct
            if "vdso" in dso or "clock_gettime" in sym or "chrono" in sym: c = "[vdso]/clock_gettime/chrono::now"
            elif "perf_adjust_freq" in sym: c = "kernel perf_adjust_freq_unthr_context (perf 자체)"
            elif dso.startswith("kt_kernel") and re.search(r"0x0*7[5-8][0-9a-f]{3}$", sym): c = "kt_kernel_ext 0x75xxx–0x78xxx (역어셈블: vpdpbusd/vpsignb/vmovdqa32 = AVX-512 VNNI)"
            elif dso.startswith("kt_kernel") and re.search(r"0x0*44[3-6][0-9a-f]{3}$", sym): c = "kt_kernel_ext 0x443xxx–0x446xxx (lock xadd 루프, unique_lock::unlock 인접)"
            elif dso.startswith("kt_kernel") and re.search(r"(unique_lock|shared_count|Sp_counted|mutex|thread)", sym): c = "kt_kernel_ext std 동기화 심볼"
            elif dso.startswith("kt_kernel"): c = "kt_kernel_ext 기타"
            elif "kernel.kallsyms" in dso: c = "kernel 기타"
            elif "libc" in dso or "libstdc" in dso: c = "libc/libstdc++ 기타"
            elif "python" in dso: c = "python"
            else: c = "기타 dso"
            cat[c] += pct
        P(f"- perf record (TP0, `-F 499`, 20 s, 298,502 샘플; 파싱된 행 합 {tot:.1f} %) 분류:")
        P("\n| 분류 | 샘플 % |\n|---|---|")
        for k, v in cat.most_common(): P(f"| {k} | {v:.2f} |")
        P("\n| 스레드(comm) 군 | 샘플 % |\n|---|---|")
        for k, v in bycomm.most_common(8): P(f"| {k} | {v:.2f} |")
        P()
    pm = f"{m2}/rep3_pcmmem/pcm_memory.csv"
    if os.path.exists(pm):
        import csv
        rows = list(csv.reader(open(pm))); g, h, data = rows[0], rows[1], rows[2:]
        def f(x):
            try: return float(x)
            except: return None
        def cols(a, pred): return [i for i in range(len(h)) if g[i] == a and pred(h[i])]
        sr = cols("System", lambda n: n == "Read"); sw = cols("System", lambda n: n == "Write")
        vals = [(sum(f(r[i]) or 0 for i in sr), sum(f(r[i]) or 0 for i in sw), r) for r in data if len(r) >= len(h)]
        busy = [v for v in vals if v[0] > 0.3 * max(x[0] for x in vals)]
        P(f"- pcm-memory (rep3, 2 s 간격, 부하 {len(busy)}/{len(vals)} 샘플): 시스템 읽기 {st.mean(v[0] for v in busy)/1000:.1f} GB/s, 쓰기 {st.mean(v[1] for v in busy)/1000:.1f} GB/s, 읽기 최대 샘플 {max(v[0] for v in busy)/1000:.1f} GB/s")
        for s in ("SKT0", "SKT1"):
            ch = cols(s, lambda n: n.endswith("Read") and "PMM" not in n and n.startswith("Ch"))
            m_ = [st.mean((f(v[2][i]) or 0) for v in busy) for i in ch]
            tr = cols(s, lambda n: n == "Mem Read (MB/s)"); tw = cols(s, lambda n: n == "Mem Write (MB/s)")
            P(f"  - {s}: 읽기 {st.mean((f(v[2][tr[0]]) or 0) for v in busy)/1000:.1f} + 쓰기 {st.mean((f(v[2][tw[0]]) or 0) for v in busy)/1000:.1f} GB/s; 채널별 읽기 MB/s {[round(x) for x in m_]}")
    P()
for m3 in all_of("*_ide070_measure3"):
    P(f"### 1.3 3차 측정 perf stat (`eval/results/{os.path.basename(m3)}/`)\n")
    for r in ("rep1_sys", "rep2_sched", "rep3_tp0"):
        b = bench(f"{m3}/{r}"); pf = f"{m3}/{r}/perf_stat.txt"
        P(f"- {r}: " + (f"{b['tput']:.2f} tok/s, TPOT p50/p95 {b['tpot']:.1f}/{b['tpot95']:.1f} ms" if b else "벤치 결과 없음 (bench.log: " + (open(f"{m3}/{r}/bench.log").read().strip()[:120] if os.path.exists(f"{m3}/{r}/bench.log") else "없음") + ")"))
        if os.path.exists(pf) and os.path.getsize(pf) > 0:
            body = "\n".join(l for l in open(pf).read().splitlines() if l.strip() and not l.startswith("#"))
            P("```"); P(body); P("```")
    P()

P("## 2. A 시리즈 — 코드 무변경 (TSK_055)\n")
P("A1 (turbo ON): `/sys/devices/system/cpu/intel_pstate/no_turbo` 쓰기가 root 로도 `Operation not permitted` (BIOS 잠금, IDE_030 2026-08-30 기록과 동일) → 미실행. A2 (물리 코어 전용): 2차 측정에서 kt 워커 96 스레드가 물리 코어 0–47, 56–103 에 고정되어 있음이 확인되어 별도 셀 없음.\n")
for b in all_of("*_ide070_aseries"): series_table("A 시리즈 셀", b, "a0 = 기준선 (cpuinfer 96, GPU 0,1,2,3); a4_cpu80/112 = cpuinfer 80/112; a6_gpu0145 = CUDA_VISIBLE_DEVICES=0,1,4,5. 각 rep 의 turbostat Bzy_MHz 는 셀 디렉터리 `rep*.turbostat.txt`.")
P("## 3. B 시리즈 — NUMA (TSK_057)\n")
for b in all_of("*_ide070_bseries"): series_table("B 시리즈 셀", b, "b1 = `numactl --interleave=all`; b2 = `numactl --membind=0`; b3 = `--kt-threadpool-count 1`; b4 = `numactl --cpunodebind=0 --membind=0` + cpuinfer 56, threadpool 1. 셀 디렉터리에 numastat_tp0.txt, tp0_thread_affinity.txt, rep*.pcm_numa.csv.")
P("## 4. C 시리즈 — hot expert 배치 (TSK_056)\n")
lb = f"{m1}/layer_budget_5952.json" if m1 else None
if lb and os.path.exists(lb):
    j = json.load(open(lb))
    P(f"- 층별 예산 (`build_layer_budget.py`, 슬롯 {j['slots']} 고정, 입력 = 1차 측정 recorder 트레이스): per_layer 최소 {j['per_layer_min']} / 최대 {j['per_layer_max']}. 같은 트레이스 기준 coverage 평균 uniform {100*j['uniform']['coverage_mean']:.3f} % → 비균일 {100*j['nonuniform']['coverage_mean']:.3f} %, coverage 최소 {100*j['uniform']['coverage_min']:.2f} % → {100*j['nonuniform']['coverage_min']:.2f} %, cold 선택/토큰 {j['uniform']['cold_selections_per_token']:.3f} → {j['nonuniform']['cold_selections_per_token']:.3f}.")
    P(f"- per_layer: `{j['per_layer']}`")
hs = f"{m1}/hotmap_v2_stats.json" if m1 else None
if hs and os.path.exists(hs):
    j = json.load(open(hs)); P(f"- hotmap_v2 (1차 측정 트레이스로 재생성, calls {j.get('total_calls', 0):,}): top-N coverage 평균/최소 = " + ", ".join(f"{n}: {100*j['coverage_mean'][n]:.1f}/{100*j['coverage_min'][n]:.1f} %" for n in ("64", "80", "96", "112", "128") if n in j.get("coverage_mean", {})))
P("- 구현: `patch_per_layer_experts.sh` 가 컨테이너의 `kt_ep_wrapper.py` `create_kt_config_from_server_args` 에서 `num_gpu_experts` 를 환경변수 `KT_GPU_EXPERTS_PER_LAYER` (json `per_layer[layer_idx]`) 로 치환. 실험 후 원복. 팩토리 단위 검증: layer0 → 142, layer33 → 75, layer61 → 102, 환경변수 없음 → 96.\n")
for b in all_of("*_ide070_tsk056"):
    nm = os.path.basename(b)
    note = "u96v2 = uniform 96 + hotmap_v2; nu5952 = 층별 예산 + hotmap_v2 + 패치."
    if any(os.path.isdir(f"{b}/nu5952") for _ in [0]) and not os.path.exists(f"{b}/nu5952/patch_log.txt"):
        pass
    pl = f"{b}/nu5952/patch_log.txt"
    if os.path.exists(pl) and open(pl).read().strip().startswith("0"):
        note += " **이 실행의 nu5952 는 패치 미적용 (patch 스크립트 `docker exec` 에 `-i` 누락, 서버 로그 per-layer 라인 0) — uniform 96 과 동일 구성.**"
    series_table(f"C 시리즈 셀 ({nm})", b, note)
P("## 5. D 시리즈 — 비동기 파이프라인 (TSK_058)\n")
P("이 빌드의 kt-kernel 에는 IDE_033 (2026-09-09) 의 callback-free 핸드오프 (mapped go/done 플래그 + 폴러 스레드 + 스트림 memop), pinned 슬롯 링 버퍼 (`KExpertsCPUBuffer`, pin_memory=True, non_blocking 복사) 가 이미 구현되어 있고 환경변수로 켠다. 셀: d1 = `KT_CALLBACK_FREE=1`; d2 = + `KT_CF_SKIP_EMPTY_IMM=1`, deferred 8; d3 = d2 + 비-kt 스레드 코어 재배치 (`pin_nonkt.sh`, 빈 코어 48-55,104-111 + HT 형제); d4 = 2026-09-10 캠페인 스택 (ENVC: KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25; hotmap_mixed_0.25; deferred 8; fp8_e5m2 KV; graph bs 32–224; max-total-tokens 143,360; chunked-prefill 4,096; mem-fraction 0.94) C64 3회 + C224 1회.\n")
for b in all_of("*_ide070_dseries"): series_table("D 시리즈 셀", b, "cf_lines.txt = 서버 로그의 `[kt-cf]` 라인 수. d2_cf_skip8: rep1 도중 요청 128/256 실패, 이후 스케줄러 프로세스 종료 (server_tail.log 에 Python fatal 덤프와 py-spy 실패 메시지만 남음), rep2·rep3 은 서버 부재로 0.")
P("## 6. E 시리즈 — GPU expert W4 (TSK_059)\n")
P("이 모델의 compressed-tensors W4A16 체크포인트는 없음 (검색: AWQ 4종, GPTQ-Int4-Int8Mix, AutoRound int4-mixed, NVFP4 6종). QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ (awq 4-bit, group 128, 252 GB) 를 사용. CPU expert 는 FP8 스냅샷에서 변환한 kt INT4 그대로. 셀: hot 96 / 128 / 144, 나머지 기준 구성.\n")
for b in all_of("*_ide070_eseries"): series_table("E 시리즈 셀", b, "부팅 실패 셀은 `server_boot_fail.log` 에 서버 로그 400행. *_nograph = `--disable-cuda-graph` 재시도.")
P("## 7. 파일 색인\n")
P("- 하네스: `eval/ide070/` (run_measure.sh, run_measure2.sh, run_measure3.sh, analyze_measure.py, build_layer_budget.py, patch_per_layer_experts.sh, run_aseries.sh, run_bseries.sh, run_tsk056.sh, run_dseries.sh, run_eseries.sh, make_result.py), 공용 `eval/ide068/lib.sh`.")
P("- 결과: `eval/results/*_ide070_*` (대용량 .pt / perf.data / speedscope 는 git 제외). 진행 로그: `shadow_assists/features/IDE_070/PROGRESS.md`. 지시서 사본: `cpu_offload_no_01_지시서.md`.")
P("- hotmap·예산: `~/.cache/huggingface/kt/ide069/hotmap.json` (v1), `~/.cache/huggingface/kt/ide070/{hotmap_v2.json, layer_budget_5952.json, hotmap_mixed_0.25.json}` (컨테이너 `/models/kt/...`).")
open(OUT, "w").write("\n".join(L) + "\n")
print(OUT, len(L), "lines")
