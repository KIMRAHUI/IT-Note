---
tags:
  - Machine_Learning
  - AI
  - Theory
  - Supervised_Learning
  - Unsupervised_Learning
created: 2026-07-29
---

#### 개요
인공지능(AI), 머신러닝(ML), 딥러닝(DL)의 포괄적 개념과 상호 관계를 명확히 정의하고, 컴퓨터가 데이터를 기반으로 가중치(Weight)를 학습하는 3단계 머신러닝 프레임워크와 주요 학습 문제 유형(지도, 비지도, 강화학습)을 정리합니다.

---

#### 인공지능 vs 머신러닝 vs 딥러닝

* **인공지능 (AI, Artificial Intelligence):** 인간의 지능을 모방하여 사고하고 문제를 해결하도록 만드는 모든 기술을 포함하는 가장 포괄적인 개념입니다.
* **머신러닝 (ML, Machine Learning):** 인공지능의 하위 분야로, 사람이 명시적인 수동 명령(Manual Instruction)을 일일이 작성하는 대신 **데이터 중심(Data-driven) 접근 방식**을 통해 통계 알고리즘으로 규칙과 패턴을 스스로 학습하는 학문입니다.
  * **톰 미첼(Tom Mitchell) 교수의 정의:** *"프로그램이 특정 작업(Task, T)을 수행하는 데 있어 경험(Experience, E)을 통해 성능(Performance, P)을 향상시키는 것"*
* **딥러닝 (DL, Deep Learning):** 머신러닝 방법론 중 하나로, 생물의 뇌 신경망 원리에 착안한 **인공신경망(Artificial Neural Network)** 기법을 활용하여 복잡한 표현을 학습합니다.

```text
┌─────────────────────────────────────────────────────────┐
│ 인공지능 (AI)                                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 머신러닝 (ML)                                      │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │ 딥러닝 (DL)                                 │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

#### 머신러닝 프레임워크 (컴퓨터는 무엇을 학습하는가?)

컴퓨터가 데이터로부터 상관관계를 파악할 때 진행되는 **3단계 학습 과정**입니다.

1. **1단계 - 가설 모델 설계 (Hypothesis):** 사람이 모델의 기본 구조나 형태(Form)를 지정합니다 (예: $y = w \cdot x + b$).
2. **2단계 - 목표 및 손실 함수 정의 (Cost/Loss Function):** 실제값과 모델 예측값 간의 오차를 최소화하는 정량적 평가 목표를 수립합니다.
3. **3단계 - 가중치 최적화 (Weight Optimization):** 학습 데이터를 사용하여 목표 오차를 가장 잘 만족시키는 최적의 가중치(Weight)를 기계가 자동으로 탐색합니다.

```text
[사람: 모델 형태(Form) 정의] ──> [데이터 입력] ──> [기계: 최적 Weight 자동 학습]
```

> **인사이트 (모델 찾기의 진짜 의미):**
> 동일한 주어진 데이터를 충족시키는 모델 형태는 무수히 많을 수 있습니다 (예: $y = 10x$ 외에도 $y = x^3 - 6x^2 + 21x - 6$ 등).
> 즉, 사람의 역할은 최적의 모델 폼(Hypothesis)을 판단하여 설정하는 것이며, 컴퓨터는 데이터로부터 미지수인 **Weight(가중치)**를 찾아내는 것입니다.

---

#### 머신러닝의 주요 학습 분야

##### ① 지도학습 (Supervised Learning)
* **특징:** 입력 데이터($X$, Feature / 요인)와 명확한 정답인 라벨($Y$, Label / Target)을 함께 제공하여 $X$와 $Y$ 사이의 관계를 학습합니다.
* **주요 종류:**
  * **회귀 (Regression):** 연속적인 실제 숫자 값을 예측하는 문제 (예: 집 크기 기반 주택 가격 예측).
  * **분류 (Classification):** 숫자로 라벨링된 카테고리(범주)를 예측하는 문제 (예: 스팸 메일 여부 분류).

##### ② 비지도학습 (Unsupervised Learning)
* **특징:** 정답 라벨($Y$)이 없이 오직 입력 데이터($X$)만 주어지며, 데이터 내부에 숨겨진 구조, 패턴, 유사성, 군집(Cluster)을 스스로 파악합니다.
* **주요 종류:** 군집화(Clustering), 차원 축소(Dimensionality Reduction).

##### ③ 강화학습 (Reinforcement Learning)
* **특징:** 정답 데이터를 직접 주지 않고, 에이전트가 환경과 상호작용하며 **보상(Reward)** 시스템을 바탕으로 시행착오를 거쳐 최적의 의사결정 액션(Action)을 선택하도록 정책을 학습합니다.

---

##### Python 코드로 보는 지도학습 vs 비지도학습 데이터 구조

```python
import pandas as pd

# 1. 지도학습 (Supervised): 입력 Feature(X)와 정답 Target(Y)이 함께 존재
supervised_data = pd.DataFrame({
    'Size_sqft': [15, 40, 20, 60],        # X (Feature)
    'Distance_km': [1, 0.5, 0.3, 2],       # X (Feature)
    'Price_100M': [3, 15, 20, 4]           # Y (Target/Label - 정답 데이터)
})

print("=== 지도학습 데이터 구조 ===")
print(supervised_data)

# 2. 비지도학습 (Unsupervised): Target(Y) 없이 입력 Feature(X)만 존재
unsupervised_data = pd.DataFrame({
    'T_shirts_count': [2, 1, 5, 0],       # X1
    'Short_skirt_count': [0, 1, 2, 4]     # X2
})

print("\n=== 비지도학습 데이터 구조 ===")
print(unsupervised_data)
```

---

#### 공식 문서 및 참고 링크

* [Scikit-learn Official Guide - Supervised & Unsupervised Learning Overview](https://scikit-learn.org/stable/user_guide.html)
* [freeCodeCamp - Machine Learning Introduction for Beginners](https://www.freecodecamp.org/news/tag/machine-learning/)
* [Google Machine Learning Crash Course - ML Concepts](https://developers.google.com/machine-learning/crash-course)
