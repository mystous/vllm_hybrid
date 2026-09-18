# IDE_076 — Claude 작업 메모
- **지시서(PLAN.md)에 있는 것만 실행한다.** 밖의 일은 이유·비용·기대를 적어 먼저 묻는다. 시각은 항상 `date` 로 확인하고 적는다.
- 컨테이너 sgl-kt: kt-kernel 소스 `/sgl-workspace/ktransformers/kt-kernel` (upstream 6d460cc + 로컬 11 파일 수정), 설치 .so `/usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so` (v4 12926df2…). 빌드: `cmake --build build/temp.linux-x86_64-cpython-312/kt_kernel.kt_kernel_ext_Release -j 48` → `build/lib.linux-x86_64-cpython-312/kt_kernel/kt_kernel_ext*.so` 를 site-packages 로 `cp`+`mv` (실행 중 프로세스 안전). 백업은 `/sgl-workspace/ide074_backup/`.
- bs=63 디코드의 Cold job 은 qlen=64 로 `forward_prefill` 경로(AMX 임계 80 미만 → `vec_mul` → `integer_mat_mul<Int4,false>` → RB `avx_kernel_rb`). `forward_decode` 는 qlen=1 전용.
- 러너 `eval/ide076/runner.py` 는 IDE_075 하네스를 importlib 로 로드해 캠페인 경로·원장·정책만 바꾼다 (횟수 guard 없음). 서버 stop 은 자기 소유 프로세스만. pkill 패턴에 자기 명령줄이 걸리지 않게 한다.
- 성능 판정은 OFF(무계측) 세션만. CORR/RESOURCE/FOCUS 는 원인 설명용.
- 커밋·푸시는 사용자 승인 후.
