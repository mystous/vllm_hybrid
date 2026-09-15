#!/usr/bin/env python3
"""IDE_023 / TSK_043 백서의 원시 데이터를 단일 MD 로 묶는다.

수록 원칙:
  - 관련 run 디렉터리의 **모든 파일을 전문 수록**한다. 발췌·요약하지 않는다.
  - 파일마다 경로·바이트·줄수·SHA256 을 함께 적는다.
  - 파생 집계(통계)는 원문과 **분리된 절**에 두고, 계산 규칙을 명시한다.
  - 원문의 제어문자(CR, ANSI)는 보이도록 치환하고 치환 사실을 적는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RES = os.path.join(ROOT, "eval", "results")
OUT = os.path.join(HERE, "IDE_023_TSK_043_RAW_DATA.md")

RUNS = [
    ("20260827_124509_tsk043_smoke_qwen30b",
     "Qwen3-30B-A3B smoke — KT 스택 검증 (t1 GPU-only / t2 KT hybrid)"),
    ("20260827_133055_tsk043_main_r1",
     "R1-0528 본판 1차 — r0 GPU-only OOM 실증 / r1 FP8 직독 실패"),
    ("20260827_140008_tsk043_main_r1",
     "R1-0528 본판 최종 — r1 AMXINT4 하이브리드 서빙"),
]

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
L: list[str] = []


def w(s: str = "") -> None:
    L.append(s)


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files_of(run: str) -> list[str]:
    base = os.path.join(RES, run)
    out = []
    for dirpath, _dirs, fs in os.walk(base):
        for f in sorted(fs):
            out.append(os.path.join(dirpath, f))
    # 상위 파일(RESULTS.md, RUN.log) 먼저, 그다음 하위 디렉터리
    return sorted(out, key=lambda p: (os.path.dirname(p) != base,
                                      os.path.dirname(p), os.path.basename(p)))


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT)


def read_text(p: str) -> tuple[str, dict]:
    """텍스트로 읽고 치환 내역을 함께 반환한다."""
    raw = open(p, "rb").read()
    info = {"bytes": len(raw)}
    try:
        s = raw.decode("utf-8")
        info["encoding"] = "utf-8"
    except UnicodeDecodeError:
        s = raw.decode("utf-8", "replace")
        info["encoding"] = "utf-8 (일부 대체문자)"
    info["cr"] = s.count("\r")
    info["ansi"] = len(ANSI.findall(s))
    s = ANSI.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n⏎ ")
    info["nul"] = s.count("\x00")
    s = s.replace("\x00", "␀")
    return s, info


def fence_for(s: str) -> str:
    """본문에 들어 있는 백틱보다 긴 fence 를 고른다."""
    n = 3
    for m in re.findall(r"`{3,}", s):
        n = max(n, len(m) + 1)
    return "`" * n


# ------------------------------------------------------------ 파생 집계
BENCH_FIELDS = [
    ("Successful requests", "완료 요청"),
    ("Failed requests", "실패 요청"),
    ("Maximum request concurrency", "최대 동시 요청"),
    ("Benchmark duration (s)", "벤치 소요 s"),
    ("Total input tokens", "입력 토큰"),
    ("Total generated tokens", "생성 토큰"),
    ("Request throughput (req/s)", "요청 처리량 req/s"),
    ("Output token throughput (tok/s)", "출력 처리량 tok/s"),
    ("Peak output token throughput (tok/s)", "순간 최대 출력 tok/s"),
    ("Peak concurrent requests", "순간 최대 동시 요청"),
    ("Total token throughput (tok/s)", "전체 처리량 tok/s"),
    ("Mean TTFT (ms)", "TTFT 평균 ms"),
    ("Median TTFT (ms)", "TTFT 중앙 ms"),
    ("P50 TTFT (ms)", "TTFT p50 ms"),
    ("P95 TTFT (ms)", "TTFT p95 ms"),
    ("Mean TPOT (ms)", "TPOT 평균 ms"),
    ("Median TPOT (ms)", "TPOT 중앙 ms"),
    ("P50 TPOT (ms)", "TPOT p50 ms"),
    ("P95 TPOT (ms)", "TPOT p95 ms"),
    ("Mean ITL (ms)", "ITL 평균 ms"),
    ("Median ITL (ms)", "ITL 중앙 ms"),
    ("P50 ITL (ms)", "ITL p50 ms"),
    ("P95 ITL (ms)", "ITL p95 ms"),
]


def parse_bench(p: str) -> dict:
    out = {}
    for line in open(p, errors="replace"):
        line = ANSI.sub("", line).split("\r")[-1].strip()
        for key, _lab in BENCH_FIELDS:
            if line.startswith(key):
                v = line[len(key):].strip(": ").strip()
                if v:
                    out[key] = v
    return out


def parse_cpu(p: str) -> list[tuple[int, float]]:
    out = []
    for line in open(p, errors="replace"):
        parts = line.split("busy=")
        if len(parts) == 2:
            try:
                out.append((int(parts[0].strip()), float(parts[1])))
            except ValueError:
                pass
    return out


def parse_gpu(p: str) -> dict:
    byg: dict[int, list[tuple[float, float, float]]] = {}
    for r in csv.reader(open(p, errors="replace")):
        if len(r) < 4:
            continue
        try:
            g = int(r[0])
            u = float(r[1].strip().rstrip(" %"))
            m = float(r[2].strip().split()[0])
            pw = float(r[3].strip().split()[0])
        except (ValueError, IndexError):
            continue
        byg.setdefault(g, []).append((u, m, pw))
    return byg


CELLS = [
    ("20260827_124509_tsk043_smoke_qwen30b", "t1_gpu_only",
     "smoke t1 — Qwen3-30B GPU-only"),
    ("20260827_124509_tsk043_smoke_qwen30b", "t2_kt_hybrid",
     "smoke t2 — Qwen3-30B KT hybrid"),
    ("20260827_140008_tsk043_main_r1", "r1_kt_hybrid",
     "r1 — R1-0528 KT hybrid (AMXINT4)"),
]


def sec_derived():
    w("## 3. 파생 집계")
    w()
    w("이 절의 값은 원시 파일에서 **계산한** 것이다. 원문 자체는 §4 이후에 전문으로 있다. "
      "계산 규칙을 각 표 아래에 적었다.")
    w()

    w("### 3.1 벤치 보고 지표 (bench.log 파싱)")
    w()
    rows = {}
    for run, cell, label in CELLS:
        p = os.path.join(RES, run, cell, "bench.log")
        if os.path.exists(p):
            rows[label] = parse_bench(p)
    labels = list(rows)
    w("| 지표 | " + " | ".join(labels) + " |")
    w("|" + "---|" * (len(labels) + 1))
    for key, lab in BENCH_FIELDS:
        vals = [rows[l_].get(key, "—") for l_ in labels]
        if all(v == "—" for v in vals):
            continue
        w(f"| {lab} (`{key}`) | " + " | ".join(vals) + " |")
    w()
    w("규칙: `bench.log` 의 `Serving Benchmark Result` 블록에서 지표명으로 시작하는 줄을 "
      "찾아 값 부분을 그대로 옮겼다. 반올림·단위 변환을 하지 않았다. "
      "`r0` 는 서버 기동 단계에서 실패해 bench.log 가 없다.")
    w()

    w("### 3.2 CPU 사용률 (cpu_util.txt)")
    w()
    w("| 셀 | 표본 | 관측 구간 s | 평균 % | 중앙 % | p90 % | 최소 % | 최대 % |")
    w("|" + "---|" * 8)
    for run, cell, label in CELLS:
        p = os.path.join(RES, run, cell, "cpu_util.txt")
        if not os.path.exists(p):
            continue
        d = parse_cpu(p)
        if not d:
            continue
        v = sorted(x[1] for x in d)
        span = d[-1][0] - d[0][0]
        w(f"| {label} | {len(d)} | {span} | {sum(v)/len(v):.2f} | "
          f"{st.median(v):.2f} | {v[int(0.9*len(v))]:.2f} | {v[0]:.2f} | "
          f"{v[-1]:.2f} |")
    w()
    w("규칙: `<unix초> busy=<백분율>` 형식의 줄만 사용. 관측 구간은 첫 표본과 마지막 "
      "표본의 시각 차이다. 워밍업 구간을 제외하지 않은 전 구간 집계다. "
      "분위수는 정렬 후 인덱스 방식(보간 없음).")
    w()

    w("### 3.3 GPU 사용률 (gpu_util.csv)")
    w()
    for run, cell, label in CELLS:
        p = os.path.join(RES, run, cell, "gpu_util.csv")
        if not os.path.exists(p):
            continue
        byg = parse_gpu(p)
        if not byg:
            continue
        w(f"**{label}** — `{rel(p)}`")
        w()
        w("| GPU | 표본 | util 평균 % | util 최대 % | HBM 평균 MiB | "
          "HBM 최대 MiB | 전력 평균 W |")
        w("|" + "---|" * 7)
        allu = []
        for g in sorted(byg):
            a = byg[g]
            u = [x[0] for x in a]
            m = [x[1] for x in a]
            pw = [x[2] for x in a]
            allu += u
            w(f"| {g} | {len(a)} | {sum(u)/len(u):.2f} | {max(u):.0f} | "
              f"{sum(m)/len(m):.0f} | {max(m):.0f} | {sum(pw)/len(pw):.1f} |")
        w(f"| **전체** | **{sum(len(v) for v in byg.values())}** | "
          f"**{sum(allu)/len(allu):.2f}** | **{max(allu):.0f}** | — | — | — |")
        w()
    w("규칙: CSV 각 행은 `<index>, <util> %, <mem> MiB, <power> W` 이며 4개 필드가 "
      "모두 파싱되는 행만 사용했다. 타임스탬프 열이 없어 벤치 구간으로 창을 좁힐 수 "
      "없다 — 전 구간(서버 기동 이후 샘플러 종료까지) 집계다.")
    w()

    w("### 3.4 CPU 사용률 전체 시계열 (r1)")
    w()
    p = os.path.join(RES, "20260827_140008_tsk043_main_r1", "r1_kt_hybrid",
                     "cpu_util.txt")
    d = parse_cpu(p)
    t0 = d[0][0]
    w("| # | unix 초 | 경과 s | busy % |")
    w("|---|---|---|---|")
    for i, (t, v) in enumerate(d, 1):
        w(f"| {i} | {t} | {t - t0} | {v:.2f} |")
    w()
    w(f"표본 {len(d)}건 전량. 원문은 해당 `cpu_util.txt` 절에 있다.")
    w()

    w("### 3.5 벤치 호출 파라미터 (bench.log 첫 줄 Namespace)")
    w()
    keys = ["backend", "base_url", "endpoint", "model", "dataset_name",
            "num_prompts", "sonnet_input_len", "sonnet_output_len",
            "sonnet_prefix_len", "max_concurrency", "request_rate", "burstiness",
            "temperature", "ignore_eos", "seed", "percentile_metrics",
            "metric_percentiles", "num_warmups", "tokenizer"]
    got = {}
    for run, cell, label in CELLS:
        p = os.path.join(RES, run, cell, "bench.log")
        if not os.path.exists(p):
            continue
        first = open(p, errors="replace").readline()
        d_ = {}
        for k in keys:
            # 따옴표 안에 콤마가 있는 값('ttft,tpot,itl')을 먼저 잡는다.
            m = re.search(rf"[ (]{re.escape(k)}=('[^']*'|\{{[^}}]*\}}|[^,)]+)", first)
            if m:
                d_[k] = m.group(1).strip()
        got[label] = d_
    labels = list(got)
    w("| 파라미터 | " + " | ".join(labels) + " |")
    w("|" + "---|" * (len(labels) + 1))
    for k in keys:
        vals = [got[l_].get(k, "—") for l_ in labels]
        if all(v == "—" for v in vals):
            continue
        vals = [v if len(v) < 70 else v[:67] + "…" for v in vals]
        w(f"| `{k}` | " + " | ".join(vals) + " |")
    w()
    w("규칙: `bench.log` 첫 줄의 `Namespace(...)` 를 정규식으로 뽑았다. 값은 원문 표기 "
      "그대로이며 70자를 넘으면 말줄임했다 (원문은 해당 절에 전문 수록).")
    w()
    w("세 셀 모두 `dataset_name='sonnet'`, `temperature=None` 이다. 후자는 "
      "`vllm bench serve` 가 더 이상 기본값으로 greedy 를 쓰지 않는다는 뜻이며, "
      "같은 로그에 경고가 함께 기록되어 있다 — 품질 판별에 쓰인 greedy 출력은 "
      "벤치가 아니라 별도 단건 호출(`smoke.json`)이다.")
    w()


# ------------------------------------------------------------ 파일 목록
def sec_inventory(inv):
    w("## 2. 파일 목록")
    w()
    w("| # | 경로 | 바이트 | 줄 | SHA256 (앞 16) | 본문 절 |")
    w("|---|---|---|---|---|---|")
    for i, (p, nsec) in enumerate(inv, 1):
        nl = sum(1 for _ in open(p, "rb"))
        w(f"| {i} | `{rel(p)}` | {os.path.getsize(p):,} | {nl:,} | "
          f"`{sha(p)[:16]}` | §{nsec} |")
    w()
    tot_b = sum(os.path.getsize(p) for p, _ in inv)
    tot_l = sum(sum(1 for _ in open(p, "rb")) for p, _ in inv)
    w(f"합계 **{len(inv)}개 파일 · {tot_b:,} 바이트 · {tot_l:,} 줄**. "
      f"전부 이 문서에 전문 수록되어 있다.")
    w()


# ------------------------------------------------------------ 본문
def main():
    w("# IDE_023 / TSK_043 — 원시 데이터 전문")
    w()
    w("CPU MoE Expert Offloading 백서 "
      "(`IDE_023_TSK_043_CPU_MoE_Offloading_백서.docx`) 가 근거로 삼은 원시 데이터 "
      "전부다. 발췌하지 않았다.")
    w()
    w("## 1. 문서 정보")
    w()
    w("| 항목 | 값 |")
    w("|---|---|")
    w("| 대상 트랙 | IDE_023 / TSK_043 — MoE Expert Offload |")
    w("| 상위 캠페인 | PLN_003 — Hybrid Regime Sweep |")
    w("| 측정일 | 2026-08-27 |")
    w("| 측정 노드 | violet-h100-016 — Xeon Platinum 8480+ ×2 (AMX) / "
      "DDR5 2 TB / H100 80 GB ×8 |")
    w("| CPU 클럭 | 2.0 GHz 고정 (turbo OFF, `no_turbo=1`) |")
    w("| 커밋 | `437c69ae5` (branch `feat/hybrid-regime-sweep`) |")
    w("| 생성 스크립트 | `shadow_assists/features/IDE_023/whitepaper/"
      "make_rawdata_md.py` |")
    w()
    w("### 수록 규칙")
    w()
    w("- 아래 3개 run 디렉터리의 **모든 파일**을 전문 수록했다. 숨김 파일"
      "(`.cpu_mon`, `.gpu_mon` — 샘플러 PID)도 포함한다.")
    w("- 파일마다 경로·바이트·줄수·SHA256 을 붙였다. 해시는 원본 바이트 기준이다.")
    w("- 파생 집계(§3)는 원문과 분리했고 계산 규칙을 명시했다.")
    w("- 원문의 ANSI 이스케이프 시퀀스는 제거했고, 캐리지 리턴(`\\r`)은 줄바꿈 + `⏎ ` "
      "로 치환했다. 진행률 표시줄이 한 줄에 겹쳐 기록된 부분을 읽을 수 있게 하기 "
      "위한 것이며, 치환 건수는 각 파일 머리에 적었다. 그 외 내용은 그대로다.")
    w("- NUL 바이트는 `␀` 로 표시한다.")
    w()
    w("### run 디렉터리")
    w()
    w("| 디렉터리 | 내용 |")
    w("|---|---|")
    for run, desc in RUNS:
        w(f"| `eval/results/{run}/` | {desc} |")
    w()
    w("---")
    w()

    # 파일 목록 + 절 번호 배정
    inv = []
    sec_no = 4
    plan = []
    for run, desc in RUNS:
        fs = files_of(run)
        for p in fs:
            inv.append((p, f"{sec_no}"))
            plan.append((sec_no, run, desc, p))
            sec_no += 1

    sec_inventory(inv)
    w("---")
    w()
    sec_derived()
    w("---")
    w()

    cur_run = None
    for nsec, run, desc, p in plan:
        if run != cur_run:
            cur_run = run
            w(f"# 원시 파일 — `eval/results/{run}/`")
            w()
            w(f"{desc}")
            w()
        s, info = read_text(p)
        nl = s.count("\n") + (0 if s.endswith("\n") else 1) if s else 0
        w(f"## {nsec}. `{rel(p)}`")
        w()
        w(f"- 바이트 **{info['bytes']:,}** · 줄 **{nl:,}** · SHA256 `{sha(p)}`")
        w(f"- 인코딩 {info['encoding']}"
          + (f" · ANSI 시퀀스 {info['ansi']}건 제거" if info["ansi"] else "")
          + (f" · CR {info['cr']}건 치환" if info["cr"] else "")
          + (f" · NUL {info['nul']}건 표시" if info["nul"] else ""))
        w()
        if p.endswith(".json"):
            try:
                obj = json.loads(open(p, errors="replace").read())
                w("정리한 형태:")
                w()
                f = "```"
                w(f + "json")
                w(json.dumps(obj, ensure_ascii=False, indent=2))
                w(f)
                w()
                w("원문:")
                w()
            except Exception:
                pass
        f = fence_for(s)
        w(f + ("text" if len(f) == 3 else ""))
        w(s.rstrip("\n") if s.strip() else "(빈 파일)")
        w(f)
        w()

    with open(OUT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"{len(L):,} 줄 / {os.path.getsize(OUT):,} 바이트 → {OUT}")
    print(f"수록 파일 {len(inv)}개")


if __name__ == "__main__":
    main()
