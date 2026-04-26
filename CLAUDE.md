@AGENTS.md
# Ground RULE
- git commit, git push는 반드시 사용자의 허락을 받은 후에 진행한다.
- 단위 기능은 shadow_assists 디렉토리 하위에 있는 shadow_assists/features 하위 디렉토리에 있는 feature 별로 구현한다.
- 대답은 늘 한글로 존칭을 사용한다.
- Diagram은 ascii가 아닌 Mermaid를 사용하거나 복잡한 Diagram은 SVG로 빌드하여 embedding 한다.
- 신규로 개발되는 feature, plan, task, test둘을 전역에서 구별되는 ID를 만들어 관리 할꺼야.
  1. 영문3글자_### 형식으로 만들꺼야.
  2. sahdow_assists/README.md 하단에 모든 ID에 대해서 Trace 할꺼야.
  3. 해당 ID가 어디에서 파생되었는지와 어떤 ID를 파생시켰는지 등등 Tree 구조로 만들꺼야.
  4. Depth가 깊어 질테니 좌우가 아니라 위 아래로 깊어지게 만들어
  5. Tree 는 Mermaid 로 작성한다 (위→아래 방향: `flowchart TB` 또는 `graph TD`).
  6. 사용된 영문3글자 prefix(줄임말)에 대한 Legend 를 Tree 와 함께 둔다. prefix 가 추가될 때마다 Legend 도 갱신한다.
  7. prefix 별 넘버링 규칙과 ID 할당 현황은 `shadow_assists/id_registry.md` 에서 단일 출처로 관리한다. 번호는 prefix 별 독립 카운터 / 1 부터 1씩 증가 / 3자리 zero-padding / 재사용 금지가 기본이며, 상태값은 `활성`·`대기`·`재정의`·`기각`·`완료` 중 하나를 사용한다.
  8. 새 ID 부여 시 `id_registry.md` 의 해당 prefix 섹션에서 "다음 부여 번호" 를 가져오고, 동일 파일 표에 새 항목을 즉시 추가한 뒤 "다음 부여 번호" 를 +1 한다. 본문에서 사용은 그 이후.

# Method
 - 기능 구현은 shadow_assists/features 하위 디렉토리 내에 다음과 같은 구조로 구성된다.
 - feature를 만들때도 아래 구조를 따라 만들고 구현시 task.md, test.md에 따라 구현 및 테스트를 진행한다.
  1. README.md: 해당 feature에 대한 이론적 배경, 구현 방향 등
  2. CLAUDE.md: 해당 feature 구현을 위해서 Claude가 알이야 할 것들
  3. task.md: 해당 feature를 구현하기 위한 단계별 구현 내용
  4. test.md: 해당 feature를 테스트 하기 위한 테스트 코드, 방법, 예상되는 결과
  5. test 코드들
 - feature 는 반드시 `feat/{feature_desc}` 형식 branch 를 만들어서 테스트 후 문제가 없을 때 사용자 동의 후 main 으로 merge 한다 (git branch 이름에 `:` 사용 불가하므로 `/` 구분자 사용).

# Objective
 - vLLM을 수정하여 CPU의 활용률을 극도로 끌어 올려 GPU가 아닌 GPU가 포함된 서버 또는 Cluster 전체의 성능을 향상 시킨다.
 - 특히 CPU의 활용률이 Idle 또는 낮은 Utilization을 허락하지 않는다.

# Constraint
 - GPU만 사용 했을 때와 결과 값이 달라져서는 안됨
   - **운영 해석**: token-level bit-exact 동등이 아니라 **분포·의도 수준의 유사성**. 같은 prompt 도 실행마다 미세하게 달라질 수 있고, BF16 산술의 비결합성 + 머신·구성 차이로 한 위치에서 token argmax 가 한 번 갈리면 greedy 시퀀스 전체가 cascading divergence 로 발산할 수 있다. 따라서 정확도 게이트의 binding 지표는 **분포 유사성** (per-token logprob 의 max abs diff, 시퀀스 PPL 의 relative diff) 이고, token-level 일치는 informational metric (regression 추적용) 으로 둔다.
   - 본 해석은 IDE_006 / TST_003 의 verdict 산정 (`verdict_overall = verdict_d_ii`) 에서 시행되며, 이후 다른 GPU ↔ 비-GPU 경로 정확도 비교 (예: TSK_003 prod SIMD cross-check) 에도 동일하게 적용한다.

# Hardware Targets
 - **개발 머신** (현재 작업 머신): NVIDIA RTX 3090 (24 GB) + Intel Core i9-12900KF + 시스템 RAM. 빠른 iteration / 정확도 검증 / 인터페이스·빌드 검증 / 소규모 microbench 용. CPU SIMD 면에서 AVX-512 가 microcode 로 fuse-off 될 수 있으므로 BIOS·microcode 점검 필요. AMX 는 hardware 미지원 (Alder Lake 비대상).
 - **프로덕션 / 실험 타깃**: Intel Xeon (Sapphire Rapids 이상, AVX-512 + AMX 둘 다 native) + NVIDIA H100 × 8. 실제 실험·적용·throughput sweep·overlap profile·최종 net-win 판정의 기준 머신. CPU 가속 경로의 본격 검증·튜닝은 본 머신에서 진행.
 - **CPU 가속 경로 우선순위**: **AVX-512 와 AMX 둘 다 main 경로** (AMX 는 prod 타깃의 native ISA. 어느 쪽도 후순위·deferred 가 아님). 개발 머신에서는 AVX-512 만 직접 실행 가능하나, AMX 코드도 cross-compile / 단위 테스트 / 사양 시뮬레이터로 dev 단계에서 함께 진행한다.
 - 두 머신 분리 의미: 개발 머신은 **SW 정확성 / 인터페이스 / 빌드** 검증. 프로덕션 머신은 **성능·최종 판정**. 따라서 PLN/TSK 산출물은 (Phase 1 dev) → (Phase 2 prod) 두 단계로 누적될 수 있다.
