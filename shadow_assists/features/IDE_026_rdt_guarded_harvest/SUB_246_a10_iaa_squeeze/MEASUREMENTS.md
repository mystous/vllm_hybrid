# SUB_246 — IAA-SQUEEZE: 압축 대역 굴절, 2026-06-15

> **판정: ⏸️ 보류 — 컨테이너에 IAA 디바이스 미노출 (하드웨어 경계 차단).**

## 명제 (변경 없음)
IAA(In-memory Analytics Accelerator)로 harvest 입력을 압축 상태로 운반/보관 →
메모리 BW 발자국 축소. 동일 유용작업당 메모리 BW ↓ 가 게이트.

## 보류 사유 (정밀)
컨테이너에서 IAA 접근 경로가 **하드웨어 경계에서 막힘**:

| 확인 항목 | 결과 |
|---|---|
| `/dev/iax` (char 디바이스) | **없음** — 컨테이너가 `--device=/dev/dsa` 만 받음 (iax 미전달) |
| `/sys/bus/iax/devices/` | **없음** — iax 버스 자체가 컨테이너에 미노출 |
| `/sys/bus/dsa/devices/` iax 노드 | 없음 (dsa0/dsa1 만) |
| `iaa_crypto`·`idxd` 커널 모듈 | **로드됨** (`/proc/modules`) → 호스트엔 IAA HW 존재 |

→ IAA HW 는 호스트에 있고 `iaa_crypto` 드라이버가 점유 중이나, **user WQ char 디바이스가
컨테이너로 전달되지 않아** ENQCMD 제출 경로 자체가 없음. DSA(SUB_236/239)는 `/dev/dsa`
가 매핑돼 가능했지만 IAA 는 불가.

## 호스트 진행 조건 (해소책)
1. 컨테이너 재기동 시 `--device=/dev/iax` 추가 매핑, **또는** 호스트에서 직접 실행.
2. 호스트에서 IAA WQ 를 **user-type shared** 로 구성 (현재 `iaa_crypto` = 커널 crypto 전용
   바인딩일 수 있음 → accel-config 로 user WQ 재구성 필요):
   ```
   sudo accel-config disable-device iax0   # iaa_crypto 점유 해제 주의 (공유 영향)
   sudo accel-config config-wq iax0/wq0.0 --mode=shared --type=user --name=squeeze \
        --group-id=0 --wq-size=16 --threshold=8
   sudo accel-config enable-device iax0 && sudo accel-config enable-wq iax0/wq0.0
   ```
3. 제출 코드는 `dsa_traffic.c`/`ferry.c` 의 ENQCMD 패턴 재사용 + IAA opcode
   (compress=0x42 / decompress=0x43, AECS 설정 descriptor 필요)로 확장.

## 비고
- DSA 두 SUB(236 채널② / 239 FERRY)가 컨테이너에서 성립했으므로, IAA 도 디바이스만
  전달되면 동일 ENQCMD 경로로 진행 가능 (코드 골격 재사용).
- 압축 대역 굴절의 *개념* 은 SUB_239 FERRY(운반 오프로드)의 연장 — "운반량 자체를 압축으로
  축소". FERRY 가 성립했으니 IAA-SQUEEZE 의 상위 가설(운반 BW 감소)은 방향성 확보.

산출물: (보류 — 호스트 IAA WQ 구성 후 진행).
