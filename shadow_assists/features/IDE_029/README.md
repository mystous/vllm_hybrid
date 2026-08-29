# IDE_029 — PlacementBound: MoE 배치의 데이터 이동 하한

고전 HPC 의 communication lower bound (Hong-Kung, Ballard-Demmel) 를 "expert 를 HBM/DRAM 어디에 두고 어디서 계산하는가" 문제로 가져온다. 기존 시스템들 (TriMoE, HybriMoE 등) 은 "우리 정책이 빠르다"를 보였고, 우리는 "**어떤 정책도 이보다 빠를 수 없다**"는 하한과, 그 하한이 규정하는 승패 경계를 제시한다. 초록·선행 delta·kill-test: `brainstorming/problem_search_20260829.md` §8. 실험 플랜: `PLN_006.md`.

배경 자산: IDE_023 캠페인의 실측 corpus (30B 15× 손해 / 480B 44.5·30.6×2 / AMX knee 43~53× / K3 분할>통합 +2~12%) 전부가 검증 데이터로 재사용된다. IDE_027 (regime atlas) 의 골격을 승계하며, 하한 형식화가 당시 기각 사유 ("확인적") 를 해소한다.
