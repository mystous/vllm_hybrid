# 2026-09-09 오후 하네스 (IDE_045~050, final sweep)

컨테이너 `sgl-kt` 의 KTransformers kt-kernel 소스에 적용한 패치 (모두 env 게이트, 기본 꺼짐; 수치 bit 동일 검증 `mb480_numcheck.py`):
- `hugepage_patch.py` + `hp_guard_fix.py` — expert 버퍼 2MB huge page (`KT_HUGEPAGE=1`), IDE_045 기각
- `amx_bcache_patch.py` — AMX INT4 GEMM B 타일 언팩 캐시 (`KT_AMX_BCACHE=1`, `KT_AMX_A_LOADD=1`), IDE_046 기각
- `qa_par_patch.py` + `qa_par_fix.py` — A 양자화 expert×32행 병렬 (`KT_QA_PAR=1`), IDE_046-b
- `avx_rb_patch.py` — AVX-512 vec 커널 레지스터 블로킹 (`KT_AVX_RB=1`), IDE_048 채택
- `avx_pf_patch.py` — rb 커널 B 스트림 SW prefetch (`KT_AVX_PF=2`), IDE_050 채택

마이크로벤치: `mb480_rows3.py` (prefill 규모), `mb480_rows_vec.py` (decode 규모 vec), `mb480_rows_pf.py`, `mb480_phase.py` (`KT_PHASE_PROF=1` 위상 분해), `mb480_decode_ht.py`.
체인: `run_*.sh` (게이트 `until grep -q <MARKER>` 로 직렬화). 최종 구성 재측정: `run_final_sweep.sh`.
