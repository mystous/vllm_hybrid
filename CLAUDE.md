@AGENTS.md
# Ground RULE
- git commit, git push는 반드시 사용자의 허락을 받은 후에 진행한다.
- 단위 기능은 shadow_assists 디렉토리 하위에 있는 shadow_assists/features 하위 디렉토리에 있는 feature 별로 구현한다.
- 대답은 늘 한글로 존칭을 사용한다.
- 신규로 개발되는 feature, plan, task, test둘을 전역에서 구별되는 ID를 만들어 관리 할꺼야.
  1. 영문3글자_### 형식으로 만들꺼야.
  2. sahdow_assists/README.md 하단에 모든 ID에 대해서 Trace 할꺼야.
  3. 해당 ID가 어디에서 파생되었는지와 어떤 ID를 파생시켰는지 등등 Tree 구조로 만들꺼야.
  4. Depth가 깊어 질테니 좌우가 아니라 위 아래로 깊어지게 만들어

# Method
 - 기능 구현은 shadow_assists/features 하위 디렉토리 내에 다음과 같은 구조로 구성된다.
 - feature를 만들때도 아래 구조를 따라 만들고 구현시 task.md, test.md에 따라 구현 및 테스트를 진행한다.
  1. README.md: 해당 feature에 대한 이론적 배경, 구현 방향 등
  2. CLAUDE.md: 해당 feature 구현을 위해서 Claude가 알이야 할 것들
  3. task.md: 해당 feature를 구현하기 위한 단계별 구현 내용
  4. test.md: 해당 feature를 테스트 하기 위한 테스트 코드, 방법, 예상되는 결과
  5. test 코드들
 - featue는 반드시 feat:{feature_desc} 로 branch를 만들어서 테스트 후 문제가 없을 때 사용자 동의 후 main으로 merge 한다.

# Objective
 - vLLM을 수정하여 CPU의 활용률을 극도로 끌어 올려 GPU가 아닌 GPU가 포함된 서버 또는 Cluster 전체의 성능을 향상 시킨다.
 - 특히 CPU의 활용률이 Idle 또는 낮은 Utilization을 허락하지 않는다.

# Constraint
 - GPU만 사용 했을 때와 결과 값이 달라져서는 안됨
