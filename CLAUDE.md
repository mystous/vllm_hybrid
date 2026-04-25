@AGENTS.md
# Ground RULE
- git commit, git push는 반드시 사용자의 허락을 받은 후에 진행함
- 단위 기능은 shadow_assists 디렉토리 하위에 있는 shadow_assists/features 하위 디렉토리에 있는 feature 별로 구현한다.

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
