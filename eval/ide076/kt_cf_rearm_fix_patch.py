#!/usr/bin/env python3
"""IDE_076 공통 정상성 수정 (사고 근본 원인): callback-free eager 패킷 rearm 경쟁.
근거: SGLang overlap 스케줄러(event_loop_overlap)는 호스트가 GPU 스트림보다 한 스텝 앞서 forward 를 발행한다. eager(비-capture) 스텝은 층별
단일 슬롯(_cf_eager_slot)을 `rearm_packet` 으로 덮어쓰므로, 이전 스텝의 go memop 이 GPU 에서 아직 발화하지 않았는데 호스트가 다음 스텝의
패킷(다른 row 수 → 다른 pinned 버퍼 집합을 가리키는 args)으로 덮어쓰면, 이전 스텝의 go 에 폴러가 다음 스텝의 args 로 작업을 넣어 아직 D2H 되지
않은 버퍼를 읽는다 → NaN. row 수가 같으면 args 가 동일해 무해하므로 '작은 prefill → 8192 청크' 전환(웨이브 시작)에서만 발현한다는 로그와 일치.
수정: (C++) 슬롯별 arm 세대/소비 세대 카운터를 두고 rearm 은 직전 go 가 폴러에 소비될 때까지 대기(GIL 해제); 폴러는 패킷 enqueue 를 마친 뒤 소비 세대를 갱신.
      (Python) 층별 eager 슬롯을 링(기본 2, env KT_CF_EAGER_RING)으로 두어 호스트 선행을 유지(대기는 거의 발생하지 않음). 이전 완화(layer 0 sync)는 제거.
컨테이너에서 apply | revert | status. 백업: /sgl-workspace/ide076_backup/{cpuinfer.h,ext_bindings.cpp}.pre_cfrearm, experts_base.py.pre_cfsync(원본)."""
import sys, os, shutil, hashlib
KT = "/sgl-workspace/ktransformers/kt-kernel"; H = f"{KT}/cpu_backend/cpuinfer.h"; EB = f"{KT}/ext_bindings.cpp"
PY = "/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; BK = "/sgl-workspace/ide076_backup"


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def rep1(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:60]); return s.replace(old, new)


def apply():
    for p, b in ((H, f"{BK}/cpuinfer.h.pre_cfrearm"), (EB, f"{BK}/ext_bindings.cpp.pre_cfrearm")):
        if not os.path.exists(b): shutil.copy(p, b)
    s = open(H).read()
    if "IDE_076 CF-rearm" in s: print("cpuinfer.h already", sha(H))
    else:
        s = rep1(s, "  unsigned slot_epoch_[kMaxPackets] = {0};\n",
                 "  unsigned slot_epoch_[kMaxPackets] = {0};\n"
                 "  // IDE_076 CF-rearm: 슬롯별 arm 세대(호스트)·go 소비 세대(폴러). rearm 은 직전 go 소비까지 대기 → 패킷 덮어쓰기 경쟁 차단\n"
                 "  unsigned cf_arm_gen_[kMaxPackets] = {0}; unsigned cf_go_gen_[kMaxPackets] = {0};\n"
                 "  std::atomic<long> n_rearm_wait_{0}, n_rearm_timeout_{0};\n")
        s = rep1(s, "    __atomic_store_n(&p.armed, true, __ATOMIC_RELEASE);\n    return slot;\n  }",
                 "    __atomic_store_n(&p.armed, true, __ATOMIC_RELEASE);\n    __atomic_store_n(&cf_arm_gen_[slot], 1u, __ATOMIC_RELEASE);   // IDE_076 CF-rearm\n    return slot;\n  }")
        s = rep1(s, "  void rearm_packet(int slot, std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def, intptr_t mask_ptr = 0, int n_experts = 0, size_t out_row_bytes = 0) {\n    Packet& p = packets_[slot];",
                 "  void rearm_packet(int slot, std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def, intptr_t mask_ptr = 0, int n_experts = 0, size_t out_row_bytes = 0) {\n"
                 "    unsigned ag = __atomic_load_n(&cf_arm_gen_[slot], __ATOMIC_ACQUIRE);   // IDE_076 CF-rearm: 직전 arm 의 go 가 폴러에 소비될 때까지 대기\n"
                 "    if (__atomic_load_n(&cf_go_gen_[slot], __ATOMIC_ACQUIRE) != ag) {\n"
                 "      n_rearm_wait_.fetch_add(1, std::memory_order_relaxed); long spins = 0;\n"
                 "      while (__atomic_load_n(&cf_go_gen_[slot], __ATOMIC_ACQUIRE) != ag) {\n"
                 "        for (int k = 0; k < 32; ++k) __builtin_ia32_pause();\n"
                 "        if (++spins == (1L << 26)) { n_rearm_timeout_.fetch_add(1, std::memory_order_relaxed); fprintf(stderr, \"[kt-cf] rearm slot=%d: previous go not consumed (arm_gen=%u go_gen=%u) after long wait; proceeding\\n\", slot, ag, __atomic_load_n(&cf_go_gen_[slot], __ATOMIC_ACQUIRE)); fflush(stderr); break; }\n"
                 "      }\n"
                 "    }\n"
                 "    Packet& p = packets_[slot];")
        s = rep1(s, "    else { p.def_fn = nullptr; p.def_args = nullptr; }\n    __atomic_store_n(&p.armed, true, __ATOMIC_RELEASE);\n  }",
                 "    else { p.def_fn = nullptr; p.def_args = nullptr; }\n    __atomic_store_n(&p.armed, true, __ATOMIC_RELEASE);\n    __atomic_store_n(&cf_arm_gen_[slot], ag + 1, __ATOMIC_RELEASE);   // IDE_076 CF-rearm\n  }")
        s = rep1(s, "        kt_evt_next_kind = 0;\n      }\n      if (!any)",
                 "        kt_evt_next_kind = 0;\n        __atomic_store_n(&cf_go_gen_[slot], __atomic_load_n(&cf_arm_gen_[slot], __ATOMIC_ACQUIRE), __ATOMIC_RELEASE);   // IDE_076 CF-rearm: 이 go 의 패킷 enqueue 완료 → 소비 세대 갱신\n      }\n      if (!any)")
        open(H, "w").write(s); print("cpuinfer.h patched", sha(H))
    e = open(EB).read()
    if "IDE_076 CF-rearm" in e: print("ext_bindings already", sha(EB))
    else:
        e = rep1(e, '      .def("rearm_packet", &CPUInfer::rearm_packet, py::arg("slot"),',
                 '      .def("rearm_packet", &CPUInfer::rearm_packet, py::call_guard<py::gil_scoped_release>() /* IDE_076 CF-rearm: 대기 중 GIL 해제 */, py::arg("slot"),')
        e = rep1(e, '      .def("arm_packet", &CPUInfer::arm_packet,',
                 '      .def("cf_rearm_stats", [](CPUInfer& c) { return py::make_tuple(c.n_rearm_wait_.load(), c.n_rearm_timeout_.load()); })   // IDE_076 CF-rearm\n      .def("arm_packet", &CPUInfer::arm_packet,')
        open(EB, "w").write(e); print("ext_bindings patched", sha(EB))
    # Python: 이전 완화(layer 0 sync) 제거 → 원본 복원 후 링 적용
    if os.path.exists(f"{BK}/experts_base.py.pre_cfsync"): shutil.copy(f"{BK}/experts_base.py.pre_cfsync", PY)
    p = open(PY).read()
    if "IDE_076 CF-rearm" in p: print("experts_base already", sha(PY))
    else:
        p = rep1(p, """                if not hasattr(self, "_cf_eager_slot"):
                    self._cf_eager_slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has, *_cf_mask_args)
                else:
                    _ci.rearm_packet(self._cf_eager_slot, _imm, _def if _has else _dummy, _has, *_cf_mask_args)
                _slot = self._cf_eager_slot
""", """                # IDE_076 CF-rearm: 층별 eager 슬롯 링(기본 2). rearm 은 C++ 에서 직전 go 소비까지 대기하므로 링은 호스트 선행(overlap 스케줄) 유지용.
                if not hasattr(self, "_cf_eager_slots"):
                    self._cf_eager_slots = []; self._cf_eager_i = 0; self._cf_eager_ring = max(1, int(_os_cf.environ.get("KT_CF_EAGER_RING", "2")))
                _ri = self._cf_eager_i % self._cf_eager_ring; self._cf_eager_i += 1
                if len(self._cf_eager_slots) <= _ri:
                    self._cf_eager_slots.append(_ci.arm_packet(_imm, _def if _has else _dummy, _has, *_cf_mask_args))
                else:
                    _ci.rearm_packet(self._cf_eager_slots[_ri], _imm, _def if _has else _dummy, _has, *_cf_mask_args)
                _slot = self._cf_eager_slots[_ri]
""")
        open(PY, "w").write(p); import ast; ast.parse(p); print("experts_base patched", sha(PY))


def revert():
    for p, b in ((H, f"{BK}/cpuinfer.h.pre_cfrearm"), (EB, f"{BK}/ext_bindings.cpp.pre_cfrearm"), (PY, f"{BK}/experts_base.py.pre_cfsync")):
        if os.path.exists(b): shutil.copy(b, p); print("reverted", p, sha(p))


def status():
    for p in (H, EB, PY): print(os.path.basename(p), sha(p), "cf-rearm" if "IDE_076 CF-rearm" in open(p).read() else ("cf-sync" if "IDE_076 CF-fix" in open(p).read() else "unpatched"))


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
