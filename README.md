# CATLM (Cat Adaptive Tiny Language Model) Python Agent & Simulator

`catlm_simulator.py`는 CATLM 개념 정의서를 기반으로 만든 레퍼런스 Python 구현입니다.

## 포함 기능

- 4개 초기 특성값(활동성/사교성/식탐/겁쟁이)과 16개 상태값 모델링
- 10개 유저 액션 + 10×16 액션-상태 영향 매트릭스 반영
- 특성값 가중치(`0.6, 0.8, 1.0, 1.3, 1.6`) 기반 상태 변화량 계산
- 위기 게이지 `Ct` 계산 및 단계(정상/주의/경고/붕괴) 판정
- 임계값 `θ`를 겁쟁이 특성에 연동
- 단순 대사 생성(톤 + 단어풀 + 이모티콘 + 누적 관계 alpha)

## 실행

```bash
python3 catlm_simulator.py
```

실행하면 데모 시나리오를 순차 적용한 로그(action, Ct, stage, dialogue)가 출력됩니다.

## 확장 포인트

- 현재 매트릭스의 정성 레이블(`+강`, `-중` 등)을 실게임 밸런스 데이터로 치환
- 단어풀을 외부 JSON/CSV 자원으로 분리해 512개 이상 확장
- 시뮬레이션 루프를 이벤트 로그 기반 배치 리플레이 방식으로 확장
- 붕괴 단계 회복 조건(BM 아이템)을 실제 게임 경제 시스템과 연동

## 관련 논문
[Link] (https://zenodo.org/records/18780274) Structural Inference Transitions Under Irreversible Survival Constraints
