# IDE_069 — Claude 작업 메모
- IDE_068/CLAUDE.md 의 노드 주의사항 전부 적용 (nerdctl 셔임, pkill 패턴, turbo OFF 불변, 캐시 경로).
- hotmap 형식은 SGLang `--init-expert-location` 이 읽는 형식을 컨테이너 소스에서 확인해 맞춘다. 추측 금지.
- deferral 은 근사 연산 — 처리량만 보고 채택하지 않는다. GSM 40 재게이트.
- IDE_030 의 재현 수치(C32 490.9)와 다르면 원인을 적는다 (컨테이너 kt-kernel post2→post1, hotmap 워크로드 차이 등).
- `sgl-kt` 는 이미지 원본과 다르다: `/sgl-workspace/ktransformers/kt-kernel` 소스 빌드(9-07)와 sglang 로컬 수정(`distributed/bootstrap.py`, `layers/attention/dsa_backend.py`, `layers/moe/kt_ep_wrapper.py`). 따라서 dual 인스턴스 컨테이너는 새 이미지에서 만들지 말고 **`docker commit sgl-kt`** 로 동일 상태를 복제한다 (`run_dual.sh` 가 수행).
