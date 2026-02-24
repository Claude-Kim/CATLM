# CATLM — Claude Code 가이드

## 프로젝트 개요

**Cat Adaptive Tiny Language Model (CATLM)**은 모바일 게임 *SNAX Cats*의 온디바이스 경량 AI 캐릭터 적응 시스템이다.
이론적 기반: *Structural Inference Transitions Under Irreversible Survival Constraints* (SIT 논문, arXiv 공개 예정).

핵심 파일:
- [catlm_simulator.py](catlm_simulator.py) — 메인 레퍼런스 구현
- [CATLM_개념_정의서.md](CATLM_개념_정의서.md) — 도메인 스펙 문서

## 실행

```bash
python3 catlm_simulator.py
```

출력 컬럼: `mode  Ct  stage  dialogue  cap  alpha  z  |A|  theta  SIT#`

## 아키텍처

### 데이터 모델

```
CatProfile   고정 특성값 — activity / sociability / appetite / cowardice (각 1~5)
CatState     가변 상태값 — 16종, 범위 0~255, clamp() 필수
CATLMAgent   에이전트 — 위 두 모델 + 시뮬레이션 루프
Mode         NORMAL | SURVIVAL
```

### 핵심 상수 (수정 시 주의)

| 상수 | 의미 | 기본값 |
|------|------|--------|
| `IMPACT_SCALE` | 레이블 → 정수 변환 (`+강`=36 … `-강`=-36) | 고정 |
| `TRAIT_MULTIPLIER` | 특성값(1~5) → 배수(0.6~1.6) | 고정 |
| `CAPACITY_DECAY_ON_CRISIS` | 위기 틱당 비가역 역량 손실 | 0.06 |
| `CAPACITY_DECAY_ON_OVERLOAD` | 피로/짜증 과부하 시 추가 손실 | 0.03 |
| `SALVATION_COOLDOWN_HOURS` | u_t 쿨다운 | 6 |
| `SALVATION_CAPACITY_SHIELD` | u_t=1 시 capacity 손실 차감 | 0.08 |
| `ATTACHMENT_GAIN` | A_t 통합 강도 기본값 | 0.015 |
| `SIT_PERSIST_K` | SIT 감지 최소 지속 틱 수 | 3 |
| `SIT_EPS` | SIT 감지 잠재 벡터 변화 임계 | 0.35 |

### tick() 실행 순서 (1 틱 = 1 시간)

```
1. t++
2. 자연 drift (배고픔+6, 심심함+5, 외로움+3×사교성배수, ...)
3. _gate_mode() → Ct vs θ → NORMAL/SURVIVAL 전환
4. _origin_salvation() → u_t (확률적, 쿨다운 제어)
5. _origin_attachment_message() → m_t → A_t
6. capacity 감소 (위기/과부하 조건, u_t=1이면 shield 적용)
7. 정책 선택: SURVIVAL → _policy_survival() / NORMAL → _policy_normal()
8. apply_action()
9. u_t=1이면 배고픔/짜증 즉각 경감
10. state.clamp()
11. salvation_cooldown 감소
12. Y_t = _survival_proxy() → _surv_stats 업데이트
13. S_t = _trust_llr_proxy() (LLR, add-1 스무딩)
14. care_alpha += η·E_t·S_t + κ·A_t  (η=0.035, κ=0.020)
15. z_t = (explore_drive, care_drive) 갱신
16. SIT 감지: 모드 전환 + |Δz| > ε + persist ≥ k → sit_events 기록
```

### 위기 게이지 Ct 계산식

```
Ct = 배고픔×0.25 + 우울×0.20 + 짜증×0.15
   + 피부악화×0.15 + 외로움×0.15×(사교성 배수)
   + 건강악화×0.10
   + 배고픔×0.03×(식탐 배수)
   + 심심함×0.05×(활동성 배수)   ← activity_bonus
```

임계값 θ: `0.5 - (겁쟁이 배수 - 1.0) × 0.15`, 범위 [0.25, 0.65]

### 비가역 역량 (capacity) 감소 구조

```
capacity: 1.0 → 0.0 (단방향 감소, 회복 없음)

capacity < 0.75 : EXPLORE, COSTUME 불가
capacity < 0.55 : TRAIN 추가 불가
capacity < 0.40 : GIFT 추가 불가
capacity < 0.25 : FEED / SNACK / PET / GROOM / IDLE / PLAY만 허용
```

## 변수 명명 규칙

| 기호 | 코드 변수 | 설명 |
|------|----------|------|
| αt | `care_alpha` | 원점 가중치 (누적 돌봄 신뢰도) |
| u_t | `u_t` (tick 내 지역) | 원점 개입 신호 (0 or 1) |
| m_t | `m_t` (tick 내 지역) | 원점 어태치먼트 메시지 |
| A_t | `A_t` (tick 내 지역) | 어태치먼트 통합 강도 |
| Y_t | `Y` (tick 내 지역) | 단기 생존 프록시 (0 or 1) |
| S_t | `S_t` (tick 내 지역) | 신뢰 LLR 프록시 |
| E_t | `E_t` (tick 내 지역) | 위기 + 개입 이벤트 플래그 |
| z_t | `self.z` | 잠재 구성 벡터 (explore_drive, care_drive) |
| Ct | 반환값 of `crisis_score()` | 위기 예보 강도 |
| θ | 반환값 of `crisis_threshold()` | 위기 발동 임계값 |

## 확장 가이드

### 새 특성 추가 시
1. `Trait` enum에 항목 추가
2. `CatProfile`에 필드 추가
3. `_trait_for_action_state()`에 매핑 규칙 추가
4. `crisis_score()` 또는 `crisis_threshold()`에 특성 가중치 반영

### 새 액션 추가 시
1. `Action` enum에 항목 추가
2. `ACTIONS_MATRIX`에 `{State: 레이블}` 딕셔너리 추가 (레이블은 `IMPACT_SCALE` 키 중 하나)
3. `_available_actions()`에서 capacity 제약 여부 결정
4. `_policy_normal()` / `_policy_survival()` 스코어링 검토

### 단어풀 확장 시
- `dialogue()` 내 `word_bank`를 외부 JSON으로 분리 후 로드
- 현재 각 톤당 3개 → 목표 512개 (개념서 스펙)

### 배치 리플레이 확장 시
- `tick()`을 이벤트 로그 기반으로 교체: `apply_action(action)` 직접 호출 후 `tick(hours=n)` 패턴 유지

## 주의 사항

- `apply_action()`은 `care_alpha`를 **갱신하지 않는다** — 갱신은 `tick()` 내에서만 이루어진다
- `_update_care_alpha()`는 현재 미사용 상태 (레거시, tick() 통합 이전 구현)
- `capacity`는 단방향 감소 — 회복 로직은 현재 미구현 (BM 아이템 연동 예정)
- `sit_events`는 `(t, old_mode, new_mode)` 튜플 리스트 — SIT 조건 충족 시에만 기록됨
- `_trust_llr_proxy()`는 add-1 스무딩 적용 — 초기 틱에서 LLR이 0에 가까운 것이 정상
- `import math`가 `_trust_llr_proxy()` 내부에 위치함 — 파일 상단으로 이동 고려 가능
