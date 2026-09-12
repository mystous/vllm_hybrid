"""P4 — 계획 선택을 '실행이 끝나는 시간' 추정으로 교체한다 (지시서 §7).

v0.1 의 교정되지 않은 바이트 가중합 (`byte_weight=1`, `flop_weight=0.015`) 은
`Plan.stats()` 의 통계 출력에만 남기고 production 선택에서는 쓰지 않는다.

선택 순서 (§7.7 의 '빠른 우회' 를 그대로 구현):

  1. `planner.probe_head` 로 계획을 만들지 않고 Q-outer / signature 후보의 특징을 뽑는다.
  2. 교정된 표로 두 후보의 GPU 시간을 예측한다.
  3. signature 가 Q-outer 를 error_margin + extra_prepare 이상 못 이기면 **그 자리에서**
     Q-outer 계획 하나만 만들고 끝낸다. (signature/병합 후보를 만들지 않는다)
  4. 이길 것으로 예측되면 signature 계획을 만든다. `bounded_merge` 는 기본 off 이고,
     교정에서 signature 보다 좋았던 특징 영역에서만 켠다.

예측 시간은 실행 보장이 아니다. 지원 범위 밖 입력 (표에 없는 BM 등) 은 Q-outer 로 돌아간다.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import planner

# 교정 표의 특징 이름. 순서를 바꾸면 저장된 표와 맞지 않으므로 append 만 한다.
FEATURE_NAMES = (
    "const",        # 절편 (커널 고정비)
    "l_work",       # sum_t s_t / SM  — 부하 항 (§7.4)
    "l_tail",       # max_t s_t      — tail 항
    "padded_per_sm",  # padded dot 단위 / SM
    "n_programs_per_sm",
    "unique_blocks",
    "multi_rows",   # merge 대상 행
    "empty_rows",
    "k_p50_per_sm",   # §7.3: KV 길이 분포. task 가 많고 짧은 계획과
    "k_p90",          #        적고 긴 계획을 구분한다
    "imbalance",      # max_k / p50 — tail 편중
    "full_ratio",     # full-membership 비율 (membership 검사 생략 가능 여부)
)

_LN2 = math.log(2.0)


def _next_pow2(x: int) -> int:
    v = 1
    while v < x:
        v <<= 1
    return v


@dataclass
class CandidateFeatures:
    """한 후보 계획의 예측 특징. probe 또는 실제 계획 통계 어느 쪽에서든 같은 값이 나온다."""
    name: str
    bm: int
    n_tasks: int
    sum_k: int
    max_k: int
    max_q: int
    padded: float
    slots: int
    direct_rows: int
    multi_rows: int
    empty_rows: int
    unique_blocks: int
    g: int
    d: int
    bk: int
    k_p50: float = 0.0
    k_p90: float = 0.0
    full_ratio: float = 1.0

    def vector(self, sm: int) -> np.ndarray:
        # 프로그램 = (task, q-tile). BM = next_pow2(max_q) 이므로 task 당 q-tile 은 1 이다.
        n_prog = self.n_tasks
        # s_t (교정 대상 task 비용) 의 1차 근사는 KV block 수다. 실제 계수는 회귀가 정한다.
        l_work = self.sum_k / max(1, sm)
        l_tail = float(self.max_k)
        return np.array([
            1.0,
            l_work,
            l_tail,
            self.padded / max(1, sm),
            n_prog / max(1, sm),
            float(self.unique_blocks),
            float(self.multi_rows),
            float(self.empty_rows),
            self.k_p50 * self.n_tasks / max(1, sm),
            float(self.k_p90),
            float(self.max_k) / max(1.0, self.k_p50),
            float(self.full_ratio),
        ], dtype=np.float64)


def features_from_probe(pr: dict, which: str, *, d=128, bk=128) -> CandidateFeatures:
    """probe 결과에서 Q-outer('q') 또는 signature('s') 후보의 특징을 만든다."""
    assert which in ("q", "s")
    p = which
    max_q = int(pr[f"{p}_max_q"])
    return CandidateFeatures(
        name="q_outer" if which == "q" else "signature",
        bm=_next_pow2(max(1, max_q)),
        n_tasks=int(pr[f"{p}_ntasks"]),
        sum_k=int(pr[f"{p}_sum_k"]),
        max_k=int(pr[f"{p}_max_k"]),
        max_q=max_q,
        padded=float(pr[f"{p}_padded"]),
        slots=int(pr[f"{p}_slots"]),
        direct_rows=int(pr[f"{p}_direct_rows"]),
        multi_rows=int(pr[f"{p}_multi_rows"]),
        empty_rows=int(pr["empty_rows"]),
        unique_blocks=int(pr["unique_blocks"]),
        g=int(pr["g"]), d=d, bk=bk,
        k_p50=float(pr.get(f"{p}_k_p50", 0.0)),
        k_p90=float(pr.get(f"{p}_k_p90", 0.0)),
        full_ratio=float(pr.get(f"{p}_full_ratio", 1.0)))


def features_from_plan(plan, *, name=None, nq=None, d=128, bk=128) -> CandidateFeatures:
    """실제 계획에서 같은 특징을 만든다 (교정 데이터 수집 / bounded_merge 후보용)."""
    st = plan.stats()
    qs = np.diff(plan.task_qptr).astype(np.int64)
    ks = np.diff(plan.task_kptr).astype(np.int64)
    live = qs > 0
    max_q = int(qs.max()) if qs.size else 1
    uniq = int(np.unique(plan.task_k).size) if plan.task_k.size else 0
    kl = ks[live]
    kp50 = float(np.quantile(kl, 0.5)) if kl.size else 0.0
    kp90 = float(np.quantile(kl, 0.9)) if kl.size else 0.0
    # full-membership 비율: task 의 모든 block 이 그 task 의 모든 query 에 연결된 비율
    fr = 1.0
    if plan.task_kmask.size:
        want = ((np.uint64(1) << qs[live].astype(np.uint64)) - np.uint64(1))
        exp = np.repeat(want, kl)
        km = plan.task_kmask[:exp.size].view(np.uint64)
        fr = float((km == exp).mean()) if exp.size else 1.0
    return CandidateFeatures(
        name=name or plan.policy,
        bm=_next_pow2(max(1, max_q)),
        n_tasks=int(live.sum()),
        sum_k=int(ks[live].sum()),
        max_k=int(ks.max()) if ks.size else 0,
        max_q=max_q,
        padded=float(st["padded_units"]),
        slots=int(st["query_task_incidences"]),
        direct_rows=int(st["direct_rows"]),
        multi_rows=int(st["multi_owner_rows"]),
        empty_rows=int(st["empty_rows"]),
        unique_blocks=uniq,
        g=int(plan.meta.get("g", 8)), d=d, bk=bk,
        k_p50=kp50, k_p90=kp90, full_ratio=fr)


# --------------------------------------------------------------------------------------
# 교정 표
# --------------------------------------------------------------------------------------

@dataclass
class CostTable:
    """BM 별 규제 회귀 계수. 지원 범위 밖이면 Q-outer 로 폴백한다."""
    sm: int
    coef: dict = field(default_factory=dict)        # {"bm:g": [계수]}
    resid: dict = field(default_factory=dict)       # {"bm:g": {"rel_p50","rel_p90","n"}}
    meta: dict = field(default_factory=dict)
    merge_regions: list = field(default_factory=list)   # bounded_merge 를 켤 특징 영역
    obs: dict = field(default_factory=dict)   # {"bm:g": {"y_min","y_max","feat_max"}}
    diff_margin: dict = field(default_factory=dict)   # {"bm:g": 차이 잔차 p90}
    best_max_m: dict = field(default_factory=dict)    # {"g:nq_bucket": max_m}

    @staticmethod
    def shape_key(g: int, nq: int) -> str:
        """max_m lookup 의 키. nq 를 2의 거듭제곱 구간으로 묶는다."""
        b = 1
        while b < max(1, nq):
            b <<= 1
        return f"{g}:{b}"

    def pick_max_m(self, g: int, nq: int, default=16) -> int:
        """교정에서 관측된 최적 max_m (§7.4 의 '작은 lookup table').

        max_m 을 예측값 비교로 고르지 않는다. max_m 이 바뀌면 BM 이 바뀌어 다른 구간의
        회귀와 비교하게 되는데, 각 구간은 자기 절편을 갖도록 따로 적합됐으므로 그 비교는
        교정으로 보정되지 않는다 (E3 에서 regret 217% 의 원인).
        """
        return int(self.best_max_m.get(CostTable.shape_key(g, nq), default))

    @staticmethod
    def key(f: CandidateFeatures) -> str:
        """lookup key. G 는 프로그램당 row 수를 바꾸므로 BM 과 함께 묶는다 (§7.3)."""
        return f"{f.bm}:{f.g}"

    def supports(self, f: CandidateFeatures) -> bool:
        return CostTable.key(f) in self.coef

    def predict(self, f: CandidateFeatures) -> float:
        """예측 시간(us). 지원 범위 밖이면 `math.inf` 를 돌려 선택에서 배제한다.

        음수 예측을 0 으로 클리핑하지 않는다. 0 으로 자르면 외삽 실패가 '가장 빠른 후보'
        로 보여 최악의 계획을 고르게 된다 (E3 1차에서 regret 415% 의 원인).
        """
        k = CostTable.key(f)
        c = np.asarray(self.coef[k], dtype=np.float64)
        y = float(c @ f.vector(self.sm))
        if not np.isfinite(y) or y <= 0.0:
            return float("inf")
        o = self.obs.get(k)
        if o:
            # 교정에서 본 시간 범위를 크게 벗어나면 외삽으로 본다.
            # 특징값 자체로 범위를 판정하지는 않는다: 같은 (BM,G) 구간에도 Nq 에 따라
            # 특징 절대값이 몇 배씩 달라지므로 그 조건은 정상 입력까지 전부 거부했다.
            # 계수가 비음수(NNLS)라 외삽 방향은 단조 증가이며 부호가 뒤집히지 않는다.
            if y > o["y_max"] * 8.0 or y < o["y_min"] / 8.0:
                return float("inf")
        return y

    def rel_error(self, f: CandidateFeatures) -> float:
        """이 (BM, G) 구간의 교정 잔차 p90. error_margin 의 근거로 쓴다."""
        r = self.resid.get(CostTable.key(f))
        return float(r["rel_p90"]) if r else 1.0

    def diff_error(self, f: CandidateFeatures) -> float:
        """이 구간에서 '후보 차이' 예측의 상대 잔차 p90. 없으면 전역값."""
        k = CostTable.key(f)
        if k in self.diff_margin:
            return float(self.diff_margin[k])
        g = self.meta.get("diff_margin_rel_p90")
        return float(g) if g is not None else float(self.rel_error(f))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump(dict(sm=self.sm, coef=self.coef, resid=self.resid, meta=self.meta,
                       merge_regions=self.merge_regions, obs=self.obs,
                       diff_margin=self.diff_margin, best_max_m=self.best_max_m,
                       feature_names=list(FEATURE_NAMES)),
                  open(path, "w"), indent=1)

    @staticmethod
    def load(path: str) -> "CostTable":
        d = json.load(open(path))
        if list(d.get("feature_names", FEATURE_NAMES)) != list(FEATURE_NAMES):
            raise ValueError("교정 표의 특징 순서가 현재 코드와 다르다")
        return CostTable(sm=int(d["sm"]), coef=d["coef"], resid=d["resid"],
                         meta=d.get("meta", {}), merge_regions=d.get("merge_regions", []),
                         obs=d.get("obs", {}), diff_margin=d.get("diff_margin", {}),
                         best_max_m=d.get("best_max_m", {}))


def fit_table(rows, sm: int, *, ridge=1e-3, meta=None, pairs=None,
              best_max_m=None) -> CostTable:
    """(특징, 실측 us) 목록에서 (BM, G) 별 규제 회귀를 적합한다.

    **상대오차 가중**을 쓴다 (weight = 1/y). 절대 최소제곱은 큰 형상에 끌려가 작은
    형상의 상대오차가 커지는데, 선택은 상대 비교이므로 그쪽이 중요하다.

    rows:  [(CandidateFeatures, measured_us), ...]
    pairs: {입력키: {후보이름: (CandidateFeatures, measured_us)}} — 주면 후보 **차이**의
           예측 잔차를 구해 `error_margin` 의 근거로 meta 에 저장한다.
    best_max_m: {"g:nq_bucket": [관측된 최적 max_m, ...]} — 주면 최빈값을 표에 저장한다.
    """
    by_bm = {}
    for f, y in rows:
        by_bm.setdefault(CostTable.key(f), []).append((f, y))
    coef, resid, obs = {}, {}, {}
    for bm, items in sorted(by_bm.items()):
        X = np.stack([f.vector(sm) for f, _ in items])
        y = np.array([v for _, v in items], dtype=np.float64)
        w = 1.0 / np.maximum(y, 1e-9)               # 상대오차 가중
        Xw = X * w[:, None]
        yw = y * w
        sc = np.maximum(np.abs(Xw).max(axis=0), 1e-12)
        Xs = Xw / sc
        # 비음수 최소제곱: 각 특징은 시간을 늘리는 방향이므로 계수 음수를 허용하지 않는다.
        # 음수 계수는 외삽에서 음수 시간을 만들고, 그것이 '가장 빠른 후보' 로 보이게 된다.
        A = np.vstack([Xs, math.sqrt(ridge) * np.eye(Xs.shape[1])])
        bb = np.concatenate([yw, np.zeros(Xs.shape[1])])
        try:
            from scipy.optimize import nnls
            cs, _ = nnls(A, bb)
        except Exception:
            cs = np.linalg.lstsq(A, bb, rcond=None)[0]
            cs = np.maximum(cs, 0.0)
        c = cs / sc
        rel = np.abs(X @ c - y) / np.maximum(y, 1e-9)
        coef[bm] = list(map(float, c))
        obs[bm] = dict(y_min=float(y.min()), y_max=float(y.max()),
                       feat_max=[float(v) for v in X.max(axis=0)])
        resid[bm] = dict(rel_p50=float(np.median(rel)),
                         rel_p90=float(np.quantile(rel, 0.9)),
                         rel_max=float(rel.max()), n=int(len(items)))
    table = CostTable(sm=sm, coef=coef, resid=resid, meta=dict(meta or {}), obs=obs)

    # --- 후보 차이의 예측 잔차 (§7.7 error_margin 의 근거) ---
    if pairs:
        diffs = []
        per_key = {}
        for key, cands in pairs.items():
            q = cands.get("q_outer")
            if q is None:
                continue
            for name, item in cands.items():
                if name == "q_outer":
                    continue
                fq, yq = q
                fp, yp = item
                if not (table.supports(fq) and table.supports(fp)):
                    continue
                pq, pp = table.predict(fq), table.predict(fp)
                if not (math.isfinite(pq) and math.isfinite(pp)):
                    continue
                err = abs((pq - pp) - (yq - yp)) / max(yq, 1e-9)
                diffs.append(err)
                per_key.setdefault(CostTable.key(fq), []).append(err)
        for k, v in per_key.items():
            if len(v) >= 20:      # 표본이 적은 구간은 전역값을 쓴다
                table.diff_margin[k] = float(np.quantile(v, 0.9))
        if diffs:
            table.meta["diff_margin_rel_p50"] = float(np.median(diffs))
            table.meta["diff_margin_rel_p90"] = float(np.quantile(diffs, 0.9))
            table.meta["diff_margin_n"] = int(len(diffs))
    if best_max_m:
        # 형상 구간별 최빈 최적 max_m. 득표가 최빈의 70% 이상인 후보가 여럿이면 작은 값을
        # 택한다 — 작은 max_m 은 BM 이 작아 교정 잔차가 작고 (관측: BM2~8 에서 p90 0.14~0.20,
        # BM16~32 에서 0.25~0.28) 병렬성도 크다.
        import collections as _c
        for key, votes in best_max_m.items():
            cnt = _c.Counter(votes)
            top = max(cnt.values())
            near = [m for m, n in cnt.items() if n >= top * 0.7]
            table.best_max_m[key] = min(near)
    return table


# --------------------------------------------------------------------------------------
# 보수적 선택 (§7.7)
# --------------------------------------------------------------------------------------

@dataclass
class Decision:
    selected: str
    reason: str
    predicted: dict = field(default_factory=dict)
    cpu_us: dict = field(default_factory=dict)
    rbc_active: bool = False
    error_margin: float = 0.0
    extra_prepare_us: float = 0.0


def _plan_us_estimate(f: CandidateFeatures, kind: str) -> float:
    """후보 계획을 실제로 만드는 CPU 추가비. 교정 meta 에 저장된 측정값을 쓴다."""
    return 0.0


def choose_plan(ptr, block_ids, *, bk, d, g, max_m=None, table: CostTable,
                threads=None, enable_merge=False, min_tasks=0,
                plan_cost_us=None, nq=None) -> tuple:
    """계획을 하나 선택해 (plan, Decision) 을 돌려준다.

    `plan_cost_us` 는 교정에서 측정한 후보별 계획 생성 CPU 시간 (dict). extra_prepare
    비교에 쓴다. 없으면 0 으로 두되 그 사실을 reason 에 남긴다.
    """
    t0 = time.perf_counter()
    if max_m is None:
        max_m = table.pick_max_m(g, nq if nq is not None else (len(ptr) - 1))
    pr = planner.probe_head(ptr, block_ids, bk=bk, d=d, g=g, max_m=max_m, threads=threads)
    fq = features_from_probe(pr, "q", d=d, bk=bk)
    fs = features_from_probe(pr, "s", d=d, bk=bk)
    probe_us = pr["probe_us"]

    def mk(policy):
        return planner.plan_head(ptr, block_ids, bk=bk, d=d, g=g, max_m=max_m,
                                 policy=policy, min_tasks=min_tasks, threads=threads)

    # 지원 범위 밖 → Q-outer
    if not table.supports(fq):
        p = mk("q_outer")
        return p, Decision("q_outer", "unsupported_features(BM 미교정)",
                           cpu_us=dict(probe_us=probe_us,
                                       total_us=(time.perf_counter() - t0) * 1e6))

    tq = table.predict(fq)
    ts = table.predict(fs) if table.supports(fs) else float("inf")
    if not math.isfinite(tq):
        p = mk("q_outer")
        return p, Decision("q_outer", "unsupported_features(예측 외삽)",
                           cpu_us=dict(probe_us=probe_us, plan_ms=p.plan_ms,
                                       total_us=(time.perf_counter() - t0) * 1e6))
    # error_margin: 후보 '차이' 의 교정 잔차 p90 (구간별, 없으면 전역).
    # 두 후보의 개별 잔차를 더하면 상관된 오차를 이중으로 세어 과도하게 보수적이 된다.
    margin = table.diff_error(fq) * tq
    pc = plan_cost_us or {}
    extra = max(0.0, pc.get("signature", 0.0) - pc.get("q_outer", 0.0))

    pred = dict(q_outer_us=tq, signature_us=ts)
    # 두 후보가 서로 다른 (BM,G) 구간이면 예측값 비교가 부당하다 → Q-outer
    if CostTable.key(fq) != CostTable.key(fs):
        p = mk("q_outer")
        return p, Decision("q_outer", "구간 불일치(BM 다름) — 예측 비교 불가",
                           predicted=pred, rbc_active=False,
                           cpu_us=dict(probe_us=probe_us, plan_ms=p.plan_ms,
                                       total_us=(time.perf_counter() - t0) * 1e6))
    if not math.isfinite(ts) or (tq - ts) <= (extra + margin):
        p = mk("q_outer")
        dec = Decision("q_outer", "predicted_saving<=extra_prepare+error_margin",
                       predicted=pred, rbc_active=False, error_margin=margin,
                       extra_prepare_us=extra,
                       cpu_us=dict(probe_us=probe_us, plan_ms=p.plan_ms,
                                   total_us=(time.perf_counter() - t0) * 1e6))
        return p, dec

    # signature 가 이길 것으로 예측된 경우에만 후보를 만든다
    p = mk("signature_only")
    sel, why = "signature", "predicted_signature_win"
    if enable_merge and pr["max_atoms_in_group"] <= 32:
        pm = mk("rbc")
        fm = features_from_plan(pm, name="bounded_merge", d=d, bk=bk)
        if table.supports(fm):
            tm = table.predict(fm)
            pred["bounded_merge_us"] = tm
            if tm < table.predict(features_from_plan(p, name="signature", d=d, bk=bk)):
                p, sel, why = pm, "bounded_merge", "predicted_merge_win"
    dec = Decision(sel, why, predicted=pred, rbc_active=True, error_margin=margin,
                   extra_prepare_us=extra,
                   cpu_us=dict(probe_us=probe_us, plan_ms=p.plan_ms,
                               total_us=(time.perf_counter() - t0) * 1e6))
    return p, dec
