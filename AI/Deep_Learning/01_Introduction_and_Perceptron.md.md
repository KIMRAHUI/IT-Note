---
date: 2026-07-29
tag: [DeepLearning, Introduction, Perceptron, History, Insight]
status: complete
---

#### 개요

인공지능(AI), 머신러닝(ML), 딥러닝(DL)의 포함 관계와 차이점을 정의하고, 가중치(Weight)를 스스로 탐색하는 머신러닝의 3단계 학습 프레임워크와 주요 학습 유형(지도, 비지도, 강화학습)을 정리합니다. 또한 딥러닝의 기초가 되는 퍼셉트론(Perceptron)의 수식 원리, 계단 함수(Step Function), 이진 크로스 엔트로피(Binary Cross-Entropy) 손실 함수를 종합적으로 다룹니다.

#### 1. 인공지능 vs 머신러닝 vs 딥러닝

- **포함 관계:** **인공지능 (AI) > 머신러닝 (ML) > 딥러닝 (Deep Learning)**
    
- **인공지능 (AI):** 인간의 지능을 모방하여 사고하고 문제를 해결하도록 만드는 모든 기술을 포함하는 가장 포괄적인 개념입니다.
    
- **머신러닝 (ML):** 사람이 수동 명령(Rule)을 작성하는 대신 "기계에게 데이터 간의 관계(패턴)를 학습시키는 방법론"입니다.
    
    - **톰 미첼(Tom Mitchell) 교수의 정의:** _"프로그램이 특정 작업(Task, T)을 수행하는 데 있어 경험(Experience, E)을 통해 성능(Performance, P)을 향상시키는 것"_
        
- **딥러닝 (DL):** 머신러닝 방법론 중 하나로, 생물의 뇌 신경망 원리에 착안하여 만들어진 **인공신경망(Neural Network) / 다층 퍼셉트론(MLP)** 기법입니다.
    


```
┌─────────────────────────────────────────────────────────┐
│ 인공지능 (AI)                                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 머신러닝 (ML)                                      │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │ 딥러닝 (DL: 인공신경망 / MLP)                │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 머신러닝 프레임워크 

컴퓨터가 데이터로부터 상관관계를 파악할 때 진행되는 **3단계 학습 과정**입니다.

1. **0단계 - 가설 모델 설계 (Hypothesis):** 사람이 모델의 기본 구조나 형태(Form, 네트워크 뼈대)를 지정합니다 (예: $y = w \cdot x + b$).
    
2. **1단계 - 목표 설정 및 초기화:** 모델의 목표 오차(Cost)를 정의하고 가중치(Weight)의 초기값을 설정합니다.
    
3. **2단계 - 오차(Cost) 확인:** 학습 데이터를 사용하여 예측 결과와 실제 값의 오차인 Cost를 확인합니다.
    
4. **3단계 - 가중치 최적화 (Weight Optimization):** 계산된 Gradient를 기반으로 가중치 업데이트 수식을 적용하여 기계가 최적의 Weight를 찾아냅니다.
    

$$W_{t+1} \leftarrow W_t - \alpha \cdot \text{Gradient}$$

- **Gradient(기울기)의 정의:** $W$의 변화량에 대한 Cost의 변화량이며, Cost가 낮아지려면 $W$가 어느 방향으로 변화해야 하는지 알려줍니다.
    


```
[사람: 모델 형태(Form) 정의] ──> [데이터 입력] ──> [기계: 최적 Weight 자동 학습]
```

> [!tip] 인사이트 (머신러닝의 본질)
> 
> 동일한 데이터를 충족시키는 모델 형태는 무수히 많을 수 있습니다 (예: $y = 10x$ 외에도 $y = x^3 - 6x^2 + 21x - 6$ 등).
> 
> 즉, **"사람이 Hypothesis(가설)의 폼을 결정하고, 기계가 Weight를 결정하는 것"**이 머신러닝의 핵심입니다.

#### 머신러닝의 주요 학습 분야

##### 지도학습 (Supervised Learning)

- **특징:** 입력 데이터($X$, Feature / 요인)와 명확한 정답인 라벨($Y$, Label / Target)을 함께 제공하여 $X$와 $Y$ 사이의 관계를 학습합니다.
    
- **주요 종류:**
    
    - **회귀 (Regression):** 연속적인 실제 숫자 값을 예측하는 문제 (예: 집 크기 기반 주택 가격 예측).
        
    - **분류 (Classification):** 숫자로 라벨링된 카테고리(범주)를 예측하는 문제 (예: 스팸 메일 여부 분류).
        

##### 비지도학습 (Unsupervised Learning)

- **특징:** 정답 라벨($Y$)이 없이 오직 입력 데이터($X$)만 주어지며, 데이터 내부에 숨겨진 구조, 패턴, 유사성, 군집(Cluster)을 스스로 파악합니다.
    
- **주요 종류:** 군집화(Clustering), 차원 축소(Dimensionality Reduction).
    

##### 강화학습 (Reinforcement Learning)

- **특징:** 정답 데이터를 직접 주지 않고, 에이전트가 환경과 상호작용하며 **보상(Reward)** 시스템을 바탕으로 시행착오를 거쳐 최적의 의사결정 액션(Action)을 선택하도록 정책을 학습합니다.
    

#### 퍼셉트론(Perceptron)의 개념과 구조 및 수식 풀이

- **어원 분석:** **Perception(지각, 인식)** + **-tron(도구, 장치)** = _"지각하는 도구"_
    
- **동작 구조 수식:**
    
    $$Y = \text{ActivateFunc}\left(\sum W_i X_i + b\right)$$
    
    - $X_1, X_2$: 입력 신호 (Input Features)
        
    - $Y$: 출력 신호
        
    - $W_1, W_2$: 가중치 (Weight) - 각 입력 신호의 중요도 조절
        
    - $b$: 바이어스 (Bias) - 모델의 발화 민감도 조절
        

> [!note] 수식 풀이 과정
> 
> 1. 각 입력 신호($X_i$)에 대응하는 가중치($W_i$)를 각각 곱합니다 ($X_1 W_1, X_2 W_2$).
>     
> 2. 이 값들을 모두 더한 뒤 모델의 민감도를 조절하는 편향($b$)을 더해 가중합을 산출합니다 ($\sum W_i X_i + b$).
>     
> 3. 이 가중합 결과를 활성화 함수($\text{ActivateFunc}$)에 통과시켜 최종 출력 $Y$를 결정합니다 (가중치가 클수록 해당 입력이 결과에 큰 영향을 미침).
>     

#### 활성화 함수 (계단 함수)와 오차 함수

##### ① 계단 함수 (Step Function)

가중합 결과가 임계값(역치) 이하이면 0, 초과이면 1을 출력하여 이진 분류를 수행합니다. SVM처럼 선형 결정 경계선을 기준으로 예측을 수행하는 성격을 가집니다.

$$y = \begin{cases} 0, & X_1 W_1 + X_2 W_2 + b \le 0 \\ 1, & X_1 W_1 + X_2 W_2 + b > 0 \end{cases}$$

##### ② 이진 크로스 엔트로피 (Binary Cross-Entropy) Cost 함수

확률적 이진 분류 모델의 오차를 정량적으로 산출하는 손실 함수입니다.

$$\text{Cost}(h_\theta(x), y) = \begin{cases} -\log(h_\theta(x)) & \text{if } y = 1 \\ -\log(1 - h_\theta(x)) & \text{if } y = 0 \end{cases}$$

- **실제값이 1일 때:** 예측 확률 $h_\theta(x)$가 1에 가까워지면 $-\log(1) = 0$이 되어 오차가 사라지고, 0에 가까워지면 $-\log(0) \to \infty$가 되어 오차가 폭발합니다.
    
- **실제값이 0일 때:** 예측 확률 $h_\theta(x)$가 0에 가까워지면 $-\log(1 - 0) = 0$이 되어 오차가 사라지고, 1에 가까워지면 $-\log(0) \to \infty$가 되어 오차가 무한대로 커집니다.
    

##### Python 코드로 보는 데이터 구조 및 퍼셉트론 구현

```
import pandas as pd
import numpy as np

# 1. 지도학습 vs 비지도학습 데이터 구조 비교
supervised_data = pd.DataFrame({
    'Size_sqft': [15, 40, 20, 60],        # X (Feature)
    'Distance_km': [1, 0.5, 0.3, 2],       # X (Feature)
    'Price_100M': [3, 15, 20, 4]           # Y (Target/Label - 정답 데이터)
})

unsupervised_data = pd.DataFrame({
    'T_shirts_count': [2, 1, 5, 0],       # X1 (Feature만 존재)
    'Short_skirt_count': [0, 1, 2, 4]     # X2
})

# 2. 퍼셉트론 동작 구현 (AND 게이트 예시)
def perceptron_AND(x1, x2):
    X = np.array([x1, x2])
    W = np.array([0.5, 0.5])  # 가중치
    b = -0.7                 # 편향 (Bias)
    
    # 가중합 계산: Sum(W * X) + b
    weighted_sum = np.sum(W * X) + b
    
    # 계단 함수(Step Function) 적용
    return 0 if weighted_sum <= 0 else 1

print("=== PERCEPTRON AND GATE ===")
print("AND(0, 0) ->", perceptron_AND(0, 0))
print("AND(1, 1) ->", perceptron_AND(1, 1))
```

####공식 문서 및 참고 링크

-  [Scikit-learn Official Guide - Supervised & Unsupervised Learning Overview](https://scikit-learn.org/stable/user_guide.html)
    
-  [Stanford CS231n - Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
    
-  [Google Machine Learning Crash Course - ML Concepts](https://developers.google.com/machine-learning/crash-course)
    
-  [freeCodeCamp - Chihuahua or Muffin? 이미지 분류 문제로 보는 머신러닝](https://www.freecodecamp.org/news/tag/machine-learning/)