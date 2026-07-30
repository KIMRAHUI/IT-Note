---
date: 2026-07-29
tag:
  - DeepLearning
  - VanishingGradient
  - Initialization
  - Dropout
  - Optimizer
  - PyTorch
  - Insight
status: complete
---

#### 개요

딥러닝 모델 학습 과정에서 직면하는 3대 주요 문제점(기울기 소실, 과적합, 속도 및 최적화)의 수학적 원리와 그에 대한 해결책(ReLU, 가중치 초기화, Dropout, Momentum)을 다룹니다. 또한 경사하강법을 발전시킨 최적화 알고리즘(Optimizer)의 수식 원리와, 프레임워크 핵심 데이터 구조인 텐서(Tensor)의 차원 개념을 함께 정리합니다.

#### 딥러닝 3대 문제점과 상세 해결책

##### ① Underfitting (기울기 소실, Gradient Vanishing)

- **원인 (수학적 풀이):**
    
    - Sigmoid 활성화 함수의 미분 최댓값은 $z=0$일 때 $\sigma'(0) = 0.25$입니다.
        
    - 역전파(Backpropagation) 과정에서 체인룰(Chain Rule)에 의해 $0 \sim 0.25$ 사이의 미분값이 층마다 지속적으로 곱해지며, 입력층에 가까워질수록 기울기(Gradient)가 $0$으로 수렴(소실)되어 가중치 학습이 진행되지 않습니다.
        
- **해결책 1: ReLU 활성화 함수 사용**
    
    - 양수 구간($z \ge 0$)에서 미분값이 항상 $1$로 유지되어 깊은 층에서도 기울기 소멸을 방지합니다.
        
        $$\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1, & z \ge 0 \\ 0, & z < 0 \end{cases}$$
        
- **해결책 2: 정밀한 가중치 초기화 (Weight Initialization)**
    
    - **모두 0으로 초기화 금지:** 모든 뉴런의 가중치를 0으로 초기화하면 모든 뉴런이 동일한 출력과 동일한 역전파 기울기를 가져 뉴런별 독립 학습(대칭성 파괴, Symmetry Breaking Failure)이 불가능해집니다.
        
    - **Xavier (Glorot) 초기화 (Sigmoid / Tanh용):** 각 노드의 출력 분산이 입력 분산과 동일하도록 가중치 분산을 조정합니다.
        
        $$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$
        
    - **He (Kaiming) 초기화 (ReLU용):** ReLU의 음수 구간($z < 0$)에서 출력이 0이 되는 현상을 고려하여 Xavier 초기화보다 분산을 2배 더 크게 설정합니다.
        
        $$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$
        

##### ② Overfitting (과적합)

- **원인:** 모델의 복잡도가 너무 높아 학습 데이터의 노이즈까지 암기하여, 보지 못한 테스트 데이터에 대한 일반화 성능이 낮아지는 현상입니다.
    
- **해결책 (Dropout):**
    
    - 학습 진행 시 레이어 내의 뉴런 중 일정 비율($p$)을 임의로 무작위 배제(0으로 차단)하여 학습시킵니다.
        
    - 특정 뉴런의 가중치에만 과도하게 의존하는 앙상블 효과를 유도하여 모델의 일반화 성능을 높입니다.
        

#### 2. 최적화 알고리즘 (Optimizers)의 발달

##### ① 기본 경사하강법 (Gradient Descent) 수식

$$W_{t+1} \leftarrow W_t - \alpha \cdot G$$

- $\alpha$: 학습률 (Learning Rate)
    
- $G$: 현재 지점에서의 기울기 ($\nabla W$)
    

##### ② Momentum (모멘텀) 수식 풀이

물리학의 관성(Momentum) 개념을 경사하강법에 도입한 최적화 방식입니다.

$$V(t) = m \cdot V(t-1) - \alpha \nabla W$$

$$W_{t+1} = W(t) + V(t)$$

- $V(t)$: 현재 시점의 속도 벡터 (Velocity)
    
- $m$: 관성 계수 (Momentum coefficient, 보통 $0.9$ 사용)
    

> [!note] 수식 해석 및 풀이
> 
> 이전 이동 방향이었던 속도 벡터 $V(t-1)$에 관성 계수 $m$을 곱해 이전 가속도를 유지한 채, 현재 지점의 기울기 감소 방향 $-\alpha \nabla W$를 더해줍니다.
> 
> 이에 따라 경사면을 따라 내려갈 때<br> **지그재그 진동(Oscillation) 현상이 대폭 완화**되고,<br> 최적점(Minimum)을 향한 수렴 속도가 비약적으로 가속화됩니다.

#### 프레임워크 구현과 텐서 (Tensor)

##### ① PyTorch 프레임워크의 역할

PyTorch는 인공신경망 모델 구조 생성, GPU 하드웨어 장치 제어(`cuda` / `mps`), 자동 미분 Engine(`autograd`), 선형대수 연산 등을 통합하여 고성능 딥러닝 애플리케이션을 구현할 수 있도록 지원합니다.

##### ② 텐서 (Tensor) 개념 및 차원 단계

다차원 데이터를 효과적으로 다루기 위한 PyTorch 및 TensorFlow의 기본 데이터 구조입니다.

- **Scalar (0D Tensor):** 단일 숫자 값 (예: `tensor(3.0)`)
    
- **Vector (1D Tensor):** 1차원 배열 (예: `tensor([1.0, 2.0, 3.0])`)
    
- **Matrix (2D Tensor):** 행과 열을 가지는 2차원 행렬 (예: `Shape (3, 4)`)
    
- **Tensor (3D+ Tensor):** 3차원 이상의 다차원 배열 (예: 이미지 데이터 `Shape (Batch, Channel, Height, Width)`)
    

##### PyTorch 코드로 보는 Dropout, Optimizer, Tensor 예시

```
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Tensor 차원 생성 예시
scalar = torch.tensor(5.0)                         # 0D Tensor
vector = torch.tensor([1.0, 2.0, 3.0])             # 1D Tensor
matrix = torch.zeros((2, 3))                        # 2D Tensor
tensor_3d = torch.randn((2, 3, 4))                 # 3D Tensor

print(f"Tensor 차원 확인: {tensor_3d.dim()}D, Shape: {tensor_3d.shape}")

# 2. He 초기화 및 Dropout을 적용한 MLP 모델 정의
class DropoutMLP(nn.Module):
    def __init__(self):
        super(DropoutMLP, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(p=0.3) # 30% 확률로 뉴런 비활성화
        self.fc2 = nn.Linear(50, 10)
        
        # He (Kaiming) 가중치 초기화 적용
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DropoutMLP()

# 3. Momentum Optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print("=== Model & Optimizer initialized successfully ===")
```

#### 공식 문서 및 참고 링크

-  [PyTorch Official Documentation - torch.optim Optimization Algorithms](https://pytorch.org/docs/stable/optim.html)
    
- [CS231n Lecture Notes - Optimization and Weight Initialization](https://cs231n.github.io/neural-networks-2/)
    
-  [TensorFlow.NET Documentation - Chapter 1: Tensor Concept](https://tensorflownet.readthedocs.io/en/latest/Tensor.html)