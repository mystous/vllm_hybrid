#!/usr/bin/env python3
"""IDE_076 §17.2 진단: NaN 발생 지점 특정 프로브 (env KT_NANPROBE=1 일 때만 동작, eager 스텝만).
층별로 (a) MoE 입력 hidden_states 의 NaN/Inf 여부, (b) CPU 출력(H2D 후 output_gpu) 의 NaN/Inf 여부, (c) rows, (d) 스텝 번호를 스트림 위에서 계산해
pinned 호스트 텐서에 non_blocking 으로 복사한다(호스트 동기화 없음 → 타이밍 보존). 백그라운드 스레드가 pinned 텐서를 5 ms 마다 읽어 첫 NaN 을
stderr 에 기록한다. 컨테이너에서 apply | revert | status. 대상 experts_base.py (백업 experts_base.py.pre_nanprobe)."""
import sys, os, shutil, hashlib
PY = "/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; BK = "/sgl-workspace/ide076_backup/experts_base.py.pre_nanprobe"
EP = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"; EPBK = "/sgl-workspace/ide076_backup/kt_ep_wrapper.py.pre_nanprobe"


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def rep1(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80]); return s.replace(old, new)


HELPER = '''
# ---- IDE_076 NaN probe (KT_NANPROBE) ----
import os as _os_np, threading as _th_np, time as _t_np
_NP_ON = bool(_os_np.environ.get("KT_NANPROBE")); _NP_LIGHT = _os_np.environ.get("KT_NANPROBE_LIGHT", "1") == "1"; _NP_MARK_RANGE = tuple(int(v) for v in _os_np.environ.get("KT_NANPROBE_MARK_LAYERS", "0-61").split("-"))
def _np_flag(t):
    mx = torch.amax(t); mn = torch.amin(t)   # NaN 은 amax/amin 으로 전파, inf 는 ±inf 로 남음
    return (torch.isnan(mx) | torch.isinf(mx) | torch.isinf(mn)).to(torch.int32)
class _KtNanProbe:
    L = 128
    def __init__(self):
        # 지연 초기화: import 시점의 pin_memory 할당은 각 rank 프로세스가 GPU 0 에 CUDA 컨텍스트를 만들어(≈520 MiB×rank) OOM 을 유발했음
        self.host = None; self.gpu_step = None; self.step = 0; self.last = {}; self.fired = 0; self.th = None
        self.on_fire = None; self._fired_keys = set()   # v4: gpu_nan 감지 시 (layer, step) 당 1 회 콜백 (kt_ep_wrapper 가 재계산 프로브 등록)
    def _ensure(self, dev):
        if self.host is None:
            self.host = torch.zeros((self.L, 8), dtype=torch.int32, pin_memory=True)   # [in_nan, out_nan, rows, step, in_bad_rows, in_first_bad_row, gpu_nan, comb_nan]
            self.gpu_step = torch.zeros((), dtype=torch.int32, device=dev)
            # 측정 커널을 별도 스트림에서 실행(forward 스트림 지연 없음 → GPU-early 타이밍 보존). 이벤트로 순서만 보장.
            self.side = torch.cuda.Stream(device=dev) if _os_np.environ.get("KT_NANPROBE_SIDE", "1") == "1" else None
            self.th = _th_np.Thread(target=self._poll, daemon=True); self.th.start()
    def _on_side(self, tensors):
        # record_stream 은 할당자가 블록 재사용을 미뤄 프리필 정점 메모리를 수 GB 키움(OOM) → 대신 forward 스트림이 '직전' side 작업 완료를 기다리게 해
        # (한 층 뒤 join; side 작업 ≈0.4 ms ≪ 층 시간이라 실제 지연 없음) 블록 재사용 순서를 보장한다.
        if self.side is None:
            return torch.cuda.current_stream(), None
        cur = torch.cuda.current_stream()
        if getattr(self, "_side_done", None) is not None: cur.wait_event(self._side_done)
        ev = torch.cuda.Event(); ev.record(cur)
        self.side.wait_event(ev)
        return self.side, ev
    def _side_finish(self):
        if self.side is not None:
            self._side_done = torch.cuda.Event(); self._side_done.record(self.side)
    def _poll(self):
        prev = None; sys.setswitchinterval(0.001)
        while True:
            _t_np.sleep(0.005)
            h = self.host
            bad = ((h[:, 0] != 0) | (h[:, 1] != 0) | (h[:, 6] != 0) | (h[:, 7] != 0)).nonzero().flatten().tolist()
            if bad and self.fired < 50:
                snap = [tuple(int(v) for v in h[l].tolist()) for l in range(self.L) if int(h[l, 2]) > 0]
                key = tuple(int(h[l, 3]) for l in bad) + tuple(bad)
                if self.on_fire is not None:
                    for l in bad:
                        k2 = (int(l), int(h[l, 3]))
                        if int(h[l, 6]) != 0 and k2 not in self._fired_keys:
                            self._fired_keys.add(k2)
                            try: self.on_fire(int(l), int(h[l, 3]))
                            except Exception as _e: print(f"[KT-NANPROBE] on_fire error {_e!r}", file=sys.stderr, flush=True)
                if key != prev:
                    prev = key; self.fired += 1
                    print(f"[KT-NANPROBE] t={_t_np.time():.3f} bad_layers={[tuple(int(v) for v in h[l].tolist()) for l in bad]} snapshot(layer:(in_nan,out_nan,rows,step,in_bad_rows,in_first_bad_row,gpu_nan,comb_nan))={[(l, sn) for l, sn in zip([l for l in range(self.L) if int(h[l, 2]) > 0], snap)]}", file=sys.stderr, flush=True)
    def mark_in(self, layer_idx, x):
        self._ensure(x.device)
        if _NP_LIGHT and layer_idx != 0:   # v6.2 light: 층 0 입력만 검사 (step 갱신은 layer 0 에서만 필요)
            return
        st, _ = self._on_side([x])
        with torch.cuda.stream(st):
            if layer_idx == 0: self.step += 1; self.gpu_step.fill_(self.step)
            f = _np_flag(x)   # 임시 텐서 없는 축소(amax/amin)만 사용 (프리필 정점 메모리 여유가 얇음)
            self.host[layer_idx, 0].copy_(f, non_blocking=True)
            rmx = torch.amax(x, dim=1); rmn = torch.amin(x, dim=1)   # 행 단위(rows 개 스칼라): NaN 행 수·첫 NaN 행
            badrow = torch.isnan(rmx) | torch.isinf(rmx) | torch.isinf(rmn)
            self.host[layer_idx, 4].copy_(badrow.sum().to(torch.int32), non_blocking=True)
            self.host[layer_idx, 5].copy_(torch.argmax(badrow.to(torch.int32)).to(torch.int32), non_blocking=True)
            self.host[layer_idx, 2].copy_(torch.full((), x.shape[0], dtype=torch.int32, device=x.device), non_blocking=True)
            self.host[layer_idx, 3].copy_(self.gpu_step, non_blocking=True)
        self._side_finish()
    def mark_out(self, layer_idx, out): self._mark_col(layer_idx, out, 1)
    def mark_gpu(self, layer_idx, t): self._mark_col(layer_idx, t, 6)     # GPU hot-expert 출력 (결합 전)
    def mark_comb(self, layer_idx, t): self._mark_col(layer_idx, t, 7)    # 결합 결과 (gpu + cpu)
    def _mark_col(self, layer_idx, t, col):
        if self.host is None or not isinstance(layer_idx, int): return
        if _NP_LIGHT and col in (1, 7): return   # light: CPU 출력·결합 결과 검사 생략
        if _NP_LIGHT and not (_NP_MARK_RANGE[0] <= layer_idx <= _NP_MARK_RANGE[1]): return   # v6.3 ultra-light: GPU 출력 검사도 캡처 층 범위로 제한
        st, _ = self._on_side([t])
        with torch.cuda.stream(st):
            f = _np_flag(t)
            self.host[layer_idx, col].copy_(f, non_blocking=True)
        self._side_finish()
_np_probe = _KtNanProbe() if _NP_ON else None
import sys
# ---- end probe ----
'''

IN_ANCHOR = "        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)\n        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)"
IN_NEW = "        if _np_probe is not None and not torch.cuda.is_current_stream_capturing(): _np_probe.mark_in(self.layer_idx, flat_hidden_states)   # IDE_076 NaN probe\n" + IN_ANCHOR
OUT_ANCHOR = "        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)\n        return output_gpu[current_slot]"
OUT_NEW = "        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)\n        if _np_probe is not None and not torch.cuda.is_current_stream_capturing(): _np_probe.mark_out(self.layer_idx, output_gpu[current_slot])   # IDE_076 NaN probe\n        return output_gpu[current_slot]"


EP_OLD2 = '        # Step 3: Execute GPU expert computation (any quantization method)\n        # This runs in parallel with CPU computation\n        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)\n'
EP_NEW2 = '        # Step 3: Execute GPU expert computation (any quantization method)\n        # This runs in parallel with CPU computation\n        if _np_probe_ep is not None and self.tp_rank == 0 and not torch.cuda.is_current_stream_capturing():   # IDE_076 NaN probe v4: 입력 복제 링 (재계산용)\n            _np_ring_put(self, layer, x, masked_topk_output, masked_dispatch_output)\n        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)\n'
EP_OLD3 = '        self.wrapper.submit_forward(\n            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream\n        )\n'
EP_NEW3 = '        if _np_probe_ep is not None and not torch.cuda.is_current_stream_capturing(): _np_ring_clone(self, x)   # IDE_076 NaN probe v5: side 스트림 복제 (forward 지연 없음)\n        self.wrapper.submit_forward(\n            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream\n        )\n'
EP_HELPER = '\n# ---- IDE_076 NaN probe v4: 입력 복제 링 + gpu_nan 감지 시 재계산 (결정적 결함 vs 일시적 경쟁 판별) ----\n_np_ring = {}\n_np_cap_range = tuple(int(v) for v in _kt_os.environ.get("KT_NANPROBE_CAPTURE_LAYERS", "0-12").split("-"))   # v6.2: 캡처 층 범위(포함) — side 트래픽 최소화\ndef _np_cap_layer(li): return _np_cap_range[0] <= li <= _np_cap_range[1]\n_np_clone_side = None\ndef _np_ring_clone(obj, x):\n    """submit 시점: side 스트림에서 x 를 복제(≈50 µs) — forward 스트림의 D2H(≈4 ms)·라우팅과 겹치므로 fused_moe 직전 대기는 실질 0."""\n    global _np_clone_side\n    li = getattr(getattr(obj, "kt_config", None), "layer_idx", None)\n    if not isinstance(li, int) or not _np_cap_layer(li): return\n    if _np_clone_side is None: _np_clone_side = torch.cuda.Stream(device=x.device)\n    cur = torch.cuda.current_stream(); ev = torch.cuda.Event(); ev.record(cur); _np_clone_side.wait_event(ev)\n    with torch.cuda.stream(_np_clone_side):\n        xc = x.clone(); done = torch.cuda.Event(); done.record(_np_clone_side)\n    _np_ring[li % 2] = {"layer_idx": li, "step": _np_probe_ep.step, "x": xc, "x_done": done}\ndef _np_ring_put(obj, layer, x, topk_output, dispatch_output):\n    li = getattr(getattr(obj, "kt_config", None), "layer_idx", None)\n    if not isinstance(li, int): return\n    e = _np_ring.get(li % 2)\n    if e is None or e["layer_idx"] != li: return\n    torch.cuda.current_stream().wait_event(e["x_done"])   # 복제가 끝난 뒤에만 in-place fused_moe 가 x 를 덮어쓰도록 (보통 이미 완료)\n    e.update({"layer": layer, "obj": obj, "ids": topk_output.topk_ids.clone(), "w": topk_output.topk_weights.clone(),\n              "topk_output": topk_output, "dispatch_output": dispatch_output._replace(hidden_states=None)})\n_np_cap = {}\ndef _np_capture(obj, out):\n    """v6: GPU 위에서 조건부 캡처 — 첫 gpu_nan 층의 (x, ids, w, layer, step, rows) 를 torch.where 로 고정 (호스트 폴링 지연 무관)."""\n    li = getattr(getattr(obj, "kt_config", None), "layer_idx", None)\n    if not isinstance(li, int) or not _np_cap_layer(li): return\n    e = _np_ring.get(li % 2)\n    if e is None or e["layer_idx"] != li or "ids" not in e: return\n    x, ids, w = e["x"], e["ids"], e["w"]; rows = x.shape[0]\n    if not _np_cap:\n        R = max(rows, int(_kt_os.environ.get("KT_NANPROBE_MAXROWS", "8192")))\n        _np_cap.update({"x": torch.zeros((R, x.shape[1]), dtype=x.dtype, device=x.device), "ids": torch.zeros((R, ids.shape[1]), dtype=ids.dtype, device=x.device),\n                        "w": torch.zeros((R, w.shape[1]), dtype=w.dtype, device=x.device), "meta": torch.zeros((4,), dtype=torch.int32, device=x.device),\n                        "flag": torch.zeros((), dtype=torch.int32, device=x.device), "flag_host": torch.zeros((), dtype=torch.int32, pin_memory=True),\n                        "side": torch.cuda.Stream(device=x.device), "by_layer": {}})\n    c = _np_cap; c["by_layer"][li] = {"layer": e["layer"], "obj": obj, "topk_output": e["topk_output"], "dispatch_output": e["dispatch_output"]}\n    if rows > c["x"].shape[0]: return\n    cur = torch.cuda.current_stream(); ev = torch.cuda.Event(); ev.record(cur); c["side"].wait_event(ev)\n    with torch.cuda.stream(c["side"]):\n        mx = torch.amax(out); mn = torch.amin(out); f = (torch.isnan(mx) | torch.isinf(mx) | torch.isinf(mn))\n        cond = f & (c["flag"] == 0)\n        torch.where(cond, x, c["x"][:rows], out=c["x"][:rows]); torch.where(cond, ids, c["ids"][:rows], out=c["ids"][:rows]); torch.where(cond, w, c["w"][:rows], out=c["w"][:rows])\n        meta = torch.tensor([li, _np_probe_ep.step, rows, 1], dtype=torch.int32, device=x.device)\n        torch.where(cond, meta, c["meta"], out=c["meta"]); c["flag"].copy_(c["flag"] | f.to(torch.int32)); c["flag_host"].copy_(c["flag"], non_blocking=True)\n        done = torch.cuda.Event(); done.record(c["side"])\n    # forward 스트림을 기다리게 하지 않는다(층당 ≈0.5 ms 지연이 경쟁을 가림). 대신 복제본을 side 작업이 끝날 때까지 붙잡아 할당자 재사용을 막는다.\n    hold = c.setdefault("hold", []); hold.append((done, x, ids, w))\n    c["hold"] = [(ev_, *t_) for (ev_, *t_) in hold if not ev_.query()]\ndef _np_on_fire(layer_idx, step):\n    import sys as _s, time as _t\n    c = _np_cap\n    if not c or int(c["flag_host"]) == 0:\n        print(f"[KT-NANPROBE-RECOMPUTE] layer={layer_idx} step={step}: no capture yet (flag_host={None if not c else int(c[\'flag_host\'])})", file=_s.stderr, flush=True); return\n    if c.get("_done"): return\n    c["_done"] = True\n    torch.cuda.synchronize(); meta = c["meta"].tolist(); li, st, rows = meta[0], meta[1], meta[2]\n    x, ids, w = c["x"][:rows], c["ids"][:rows], c["w"][:rows]; bl = c["by_layer"].get(li, {})\n    def _bad_rows(t):\n        mx = torch.amax(t, dim=1); mn = torch.amin(t, dim=1); return (torch.isnan(mx) | torch.isinf(mx) | torch.isinf(mn))\n    xb = _bad_rows(x); wb = torch.isnan(w).any(dim=1) | torch.isinf(w).any(dim=1)\n    info = {"captured_layer": li, "captured_step": st, "rows": rows, "x_bad_rows": int(xb.sum()), "x_absmax": float(x.float().abs().max()), "w_bad_rows": int(wb.sum()), "ids_min": int(ids.min()), "ids_max": int(ids.max())}\n    try: torch.save({"layer_idx": li, "step": st, "x": x.cpu(), "ids": ids.cpu(), "w": w.cpu()}, f"/tmp/ide076_nan_dump_L{li}_S{st}.pt")\n    except Exception as _e: info["dump_error"] = repr(_e)\n    res = []\n    try:\n        for rep in range(2):\n            outs = []\n            for c0 in range(0, rows, 1024):   # 1,024 행 청크로 재계산 (메모리)\n                do = bl["dispatch_output"]._replace(hidden_states=x[c0:c0 + 1024].clone(), topk_output=bl["topk_output"]._replace(topk_ids=ids[c0:c0 + 1024].clone(), topk_weights=w[c0:c0 + 1024].clone()))\n                o = bl["obj"].gpu_method.apply(bl["layer"], do).hidden_states\n                outs.append(_bad_rows(o).nonzero().flatten() + c0)\n            torch.cuda.synchronize(); bad = torch.cat(outs) if outs else torch.zeros(0)\n            res.append({"recompute_bad_rows": int(bad.numel()), "first_bad_rows": bad[:8].tolist()})\n    except Exception as _e: res.append({"recompute_error": repr(_e)})\n    print(f"[KT-NANPROBE-RECOMPUTE] t={_t.time():.3f} fired_for=({layer_idx},{step}) captured={info} recompute={res}", file=_s.stderr, flush=True)\nif _np_probe_ep is not None: _np_probe_ep.on_fire = _np_on_fire\n# ---- end v4 ----\n'
EP_OLD = '            _rsf = getattr(getattr(layer, "moe_runner_config", None), "routed_scaling_factor", None)   # IDE_075_RSF: GPU 러너가 hot 기여에 곱하는 routed_scaling_factor 를 CPU 기여에도 적용 (None/1.0 → 무변경)\n            output = output + (cpu_output * _rsf if (_rsf is not None and _rsf != 1.0) else cpu_output)\n'
EP_NEW = '            _rsf = getattr(getattr(layer, "moe_runner_config", None), "routed_scaling_factor", None)   # IDE_075_RSF: GPU 러너가 hot 기여에 곱하는 routed_scaling_factor 를 CPU 기여에도 적용 (None/1.0 → 무변경)\n            if _np_probe_ep is not None and not torch.cuda.is_current_stream_capturing(): _np_probe_ep.mark_gpu(getattr(getattr(self, "kt_config", None), "layer_idx", "?"), output)   # IDE_076 NaN probe\n            if _np_probe_ep is not None and not torch.cuda.is_current_stream_capturing(): _np_capture(self, output)   # IDE_076 NaN probe v6: 조건부 캡처\n            output = output + (cpu_output * _rsf if (_rsf is not None and _rsf != 1.0) else cpu_output)\n            if _np_probe_ep is not None and not torch.cuda.is_current_stream_capturing(): _np_probe_ep.mark_comb(getattr(getattr(self, "kt_config", None), "layer_idx", "?"), output)   # IDE_076 NaN probe\n'


def apply():
    if not os.path.exists(EPBK): shutil.copy(EP, EPBK)
    e = open(EP).read()
    if 'IDE_076 NaN probe' not in e:
        e = rep1(e, EP_OLD, EP_NEW); e = rep1(e, EP_OLD2, EP_NEW2); e = rep1(e, EP_OLD3, EP_NEW3)
        e = rep1(e, '_KT_NAN_PROBE = _kt_os.environ.get("KT_NAN_PROBE", "0") == "1"', '_KT_NAN_PROBE = _kt_os.environ.get("KT_NAN_PROBE", "0") == "1"\ntry:\n    from kt_kernel.experts_base import _np_probe as _np_probe_ep   # IDE_076 NaN probe\nexcept Exception:\n    _np_probe_ep = None' + EP_HELPER)
        import ast; ast.parse(e); open(EP, 'w').write(e); print('kt_ep_wrapper patched', sha(EP))
    if not os.path.exists(BK): shutil.copy(PY, BK)
    s = open(PY).read()
    if "IDE_076 NaN probe" in s: print("already", sha(PY)); return
    i = s.index("\nclass KExpertsCPUBuffer"); s = s[:i] + "\n" + HELPER + s[i:]
    s = rep1(s, IN_ANCHOR, IN_NEW); s = rep1(s, OUT_ANCHOR, OUT_NEW)
    import ast; ast.parse(s); open(PY, "w").write(s); print("applied", sha(PY))


def revert():
    if os.path.exists(BK): shutil.copy(BK, PY); print("reverted", sha(PY))
    if os.path.exists(EPBK): shutil.copy(EPBK, EP); print("reverted ep", sha(EP))


def status(): print("experts_base.py", sha(PY), "nanprobe" if "IDE_076 NaN probe" in open(PY).read() else "no-probe")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
