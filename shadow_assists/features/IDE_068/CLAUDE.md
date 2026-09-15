# IDE_068 — Claude 작업 메모

- 노드는 violet-h100-016 (prod 급). CLAUDE.md 의 "개발 머신 = RTX 3090" 은 이 노드에 해당하지 않는다.
- `~/bin/docker` = `sudo nerdctl` 셔임. 컨테이너는 `--net host`. sudo 는 NOPASSWD.
- 컨테이너 `sgl-kt`: `~/.cache/huggingface` (= `/data/mystous/.cache/huggingface`, 셸 `HF_HOME`) → `/models`. kt 변환본은 `/models/kt/*`. Kimi-K2 도 같은 캐시로 받으므로 추가 마운트 불필요.
- SGLang 은 480B FP8 를 TP=8 순수 분할 못 함 (`output_size 320 not divisible by block_n=128`) → GPU-only 기준선은 `--tp 8 --ep-size 8`.
- `pkill -f` 는 반드시 `patter[n]` 형태 (self-match 자살 사고 3회).
- turbo OFF 는 사용자 결정 사항. 해제하지 않는다.
- 다운로드 대역폭 실측 39.5 MB/s (단일 스트림). `hf download --max-workers 16` 으로 병렬.
- push 는 이 실험(IDE_068)에 한해 사용자 승인. 다른 미커밋 파일(EWP 등)은 섞지 않는다.
