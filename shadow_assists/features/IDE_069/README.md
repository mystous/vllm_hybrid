# IDE_069 — 480B TP4 하이브리드 처리량 향상: HBM 추가 사용 + hot expert 배치

> 부모 `IDE_068` (실험 B). 작업 `TSK_052`, 게이트 `TST_026`. 브랜치 `feat/480b-tp4-hot-expert`.
> 노드 violet-h100-016 — H100 80 GB ×8 / Xeon 8480+ ×2 (AMX) / DDR5 2 TB / **turbo OFF 2.0 GHz 고정 (모든 CPU 수치는 하한)**.
> 사용자 지시 (2026-09-15): "GPU 4개를 사용한 480B 실험의 성능을 GPU HBM 추가 사용 및 hot expert 배치를 통해서 향상시킬 수 있을 것 같아. 다양한 시도를 통해서 metric 을 향상시켜 봐."

## 출발점과 참조

| | 구성 | C16 | C32 | 출처 |
|---|---|---|---|---|
| 출발점 | TP4, expert 전량 CPU, graph OFF, HBM 17 GiB/장 | **43.39** | — | IDE_068 b1 |
| 물리 id 96개 | TP4 + `--kt-num-gpu-experts 96` (hotmap 없음), graph OFF | 51.55 | — | IDE_068 b6 |
| IDE_030 처방 | TP4 + hotmap 96 + graph ON + deferral 4 + dispatch dynamic | — | **490.9** | IDE_030 (2026-09-08), GSM100 97 % |
| IDE_030 spec | 위 + STANDALONE spec (Qwen3-4B draft) | — | 1.45~1.55× | IDE_030 (2026-08-30) |

IDE_030 의 hotmap.json 은 소실됐다. 재생성이 첫 단계다.

## 손잡이 (효과 크기 순)

1. **hotmap** — 라우팅 빈도 상위 expert 를 물리 id 앞쪽으로 재배열 → `--kt-num-gpu-experts N` 이 실제 hot 을 잡게 함. IDE_068 b6 (물리 id 96, +19 %) 대 IDE_030 (hotmap 96, +1,000 %) 의 차이가 이것.
2. **hot expert 수** — HBM 여유 장당 ≈60 GB. 96개 = 장당 70 GB (mem-fraction 0.92). 112개는 +11.8 GB/장 필요 → KV 축소 또는 mem-fraction 0.95.
3. **cuda graph** — expert 가 GPU 에 충분히 올라가면 GPU 구간이 스텝의 주요 부분이 되어 launch overhead 가 드러난다 (IDE_068 b7/b8 에서 expert 0·16 일 때는 효과 없음).
4. **deferral** (`--kt-max-deferred-experts-per-token N`) — 토큰별 가중치 하위 N expert 의 CPU 계산을 다음 층과 겹침. 근사 → 품질 재게이트 필수.
5. **동시성** — CPU expert 는 배치가 클수록 행 병합 이득. C16/32/64 sweep.
6. **dispatch dynamic** — `--ep-dispatch-algorithm dynamic`.
7. **spec decode** — 별도 표기 (GPU 측 손잡이).

## 원칙

성립 = 전 요청 완료 + greedy 4/4. deferral > 0 셀은 GSM 40문항이 deferral 0 셀 대비 −2 문항 이내여야 채택. CPU busy 는 보조. 원본은 `eval/results/<TS>_ide069_*/`.
