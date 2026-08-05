---
date: 2026-07-29
tag: [DeepLearning, LossFunction, CrossEntropy, Softmax, ParameterCount, Insight]
status: complete
---

#### 개요

다층 퍼셉트론(MLP)의 층간 연결 구조에 따른 **가중치(Weight) 및 편향(Bias) 파라미터 수 계산 원리**와 실전 계산 과정을 정리합니다. 또한 다중 클래스 분류 문제를 다룰 때 출력값을 확률 분포로 정규화하는 **Softmax 함수**와, 정답 클래스의 예측 확률만을 평가하는 **Cross Entropy Loss 수식 풀이**를 다룹니다.

#### 다층 퍼셉트론의 Weight 및 Bias 개수 계산 풀이

인공신경망의 전체 학습 파라미터(Trainable Parameters) 수는 입력층, 은닉층, 출력층의 노드 수와 연결 방식에 의해 결정됩니다.

##### 기본 계산 공식

- **가중치(Weight) 개수:** $(\text{이전 층 노드 수}) \times (\text{다음 층 노드 수})$
    
- **편향(Bias) 개수:** 입력층을 제외한 **모든 다음 층(은닉층 + 출력층) 노드 수의 총합**
    

#### 실전 계산 예시 (입력 100개, 은닉층 100개짜리 29개 층, 출력 3개)

##### 1) 가중치 (Weight) 계산

1. **입력층 $\rightarrow$ 첫 번째 은닉층:**
    
    $$100 \times 100 = 10,000 \text{개}$$
    
2. **은닉층 $\rightarrow$ 은닉층 (총 29개 은닉층 중 나머지 28개 구간):**
    
    $$100 \times 100 \times 28 = 280,000 \text{개}$$
    
3. **마지막 은닉층 $\rightarrow$ 출력층:**
    
    $$100 \times 3 = 300 \text{개}$$
    

$$\text{총 Weight 개수} = 10,000 + 280,000 + 300 = 290,300 \text{개}$$

##### 2) 편향 (Bias) 계산

- **은닉층 편향 총합:** $100 \text{개} \times 29 \text{개 층} = 2,900 \text{개}$
    
- **출력층 편향:** $3 \text{개}$
    

$$\text{총 Bias 개수} = (100 \times 29) + 3 = 2,903 \text{개}$$

##### 3) 최종 학습 파라미터 수

$$\text{최종 파라미터 합} = 290,300 + 2,903 = 293,203 \text{개}$$

#### 손실 함수 (Loss Function)와 교차 엔트로피 수식 풀이

##### ① 다중 클래스 Softmax 함수 풀이

Softmax 함수는 모델의 마지막 출력층에서 나온 점수(Raw Score, Logit $z$)들을 입력받아 모든 클래스 확률의 합이 정확히 $1$ ($100\%$)이 되도록 변환하는 다중 분류용 정규화 활성화 함수입니다.

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}} \quad (\text{for } i = 1, 2, \dots, k)$$

- Exponential($e^z$)을 취해 음수 점수를 모두 양수로 변환하고 큰 점수의 격차를 부각시킵니다.
    
- 전체 합으로 나누어 각 클래스별 발생 확률 $p_i$ ($0 \le p_i \le 1$)를 구합니다.
    

##### ② 다중 클래스 교차 엔트로피 (Cross Entropy) 손실 공식 풀이

Softmax를 거쳐 나온 예측 확률 분포와 실제 정답(One-hot Vector) 사이의 오차를 계산하는 손실 함수입니다.

$$\text{Loss} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

- $C$: 전체 클래스 개수
    
- $y_c$: 정답 라벨의 원-핫 인코딩 값 ($1$ 또는 $0$)
    
- $\hat{y}_c$: 모델이 예측한 클래스 $c$일 확률 ($0 \sim 1$)
    

> [!note] 수식 해석 방식
> 
> 실제 정답이 아닌 오답 클래스들은 $y_c = 0$이 되어 앞에 $0$이 곱해지므로 계산에서 완전히 제외됩니다.
> 
> 즉, **오직 정답 위치($y_c = 1$)의 예측 확률 $\hat{y}_c$만 손실 연산에 반영**됩니다.
> 
> - 정답 확률 $\hat{y}_c \to 1$ 인 경우: $-\log(1) = 0$ (손실 없음)
>     
> - 정답 확률 $\hat{y}_c \to 0$ 인 경우: $-\log(0) \to \infty$ (손실 폭발)
>     

##### PyTorch 코드로 보는 파라미터 수 계산 및 Cross Entropy 연산 예시

```
import torch
import torch.nn as nn

# 1. 모델 정의 (입력 100, 은닉 100x29층, 출력 3)
layers = [nn.Linear(100, 100)]
for _ in range(28):
    layers.append(nn.ReLU())
    layers.append(nn.Linear(100, 100))
layers.append(nn.ReLU())
layers.append(nn.Linear(100, 3))

model = nn.Sequential(*layers)

# 파라미터 총 개수 계산
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"=== 모델 파라미터 총 개수: {total_params:,} 개 ===")

# 2. Cross Entropy Loss 연산 예시
loss_fn = nn.CrossEntropyLoss()

# Logits (예측값 raw score 2개 샘플, 3개 클래스)
logits = torch.tensor([2.0, 1.0, 0.1], [0.5, 3.0, 0.2](./2.0, 1.0, 0.1], [0.5, 3.0, 0.2.md))
# 정답 클래스 인덱스 (첫 번째는 0번 클래스, 두 번째는 1번 클래스가 정답)
targets = torch.tensor([0, 1])

loss = loss_fn(logits, targets)
print(f"Calculated Cross Entropy Loss: {loss.item():.4f}")
```

#### 공식 문서 및 참고 링크

-  [PyTorch Official Documentation - torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    
-  [Scikit-Learn User Guide - Loss, Performance and Optimization](https://scikit-learn.org/stable/modules/model_evaluation.html)
    
-  [Deep Learning Book (Goodfellow et al.) - Softmax Units for Multinoulli Output Distributions](https://www.deeplearningbook.org/contents/mlp.html)