---
date: 2026-07-29
tag: [DeepLearning, MLP, XOR, ActivationFunction, Insight]
status: complete
---

#### 개요

단층 퍼셉트론의 구조적 한계인 XOR 문제(선형 분리 불가)를 살펴보고, 은닉층을 여러 개 쌓더라도 비선형 활성화 함수가 없으면 단일 선형 모델과 동일해진다는 수학적 증명을 다룹니다. 또한 이를 해결하기 위한 주요 비선형 활성화 함수(Sigmoid, Tanh, ReLU, LeakyReLU)의 정의, 수식 및 미분 공식을 정리합니다.

#### 퍼셉트론의 한계와 XOR 문제

- **선형 분리의 한계:** 단층 퍼셉트론은 평면 위에 하나의 직선(선형 결정 경계선)을 그어 영역을 나누는 모델입니다.
    
- **논리 게이트 비교:**
    
    - **AND, OR, NAND:** 단일 직선 하나로 0과 1의 클래스를 완전히 분리할 수 있습니다.
        
    - **XOR (Exclusive OR):** 입력값이 서로 다를 때만 1을 출력하는 구조로, 단일 직선만으로는 절대 0과 1을 분리할 수 없는 **비선형 분리 문제**를 가집니다.
        


```
  [AND / OR 게이트: 선형 분리 가능]         [XOR 게이트: 선형 분리 불가능]
       y                                       y
       │   o                                   │   x       o
       │       o                               │
  ─────┼─────────── x                     ─────┼─────────── x
       │   x   x                               │   o       x
       │                                       │
```

#### 다층 퍼셉트론(MLP)과 선형 함수 합성의 수학적 증명

단층 퍼셉트론의 한계를 극복하기 위해 은닉층(Hidden Layer)을 추가한 다층 퍼셉트론(MLP)을 사용합니다. 이때 은닉층 사이에는 반드시 비선형 활성화 함수(Non-linear Activation Function)가 추가되어야 합니다.

> [!note] 선형 함수 합성의 한계 증명
> 
> 두 개의 선형 함수 $f(x) = W_f x + b_f$ 와 $g(x) = W_g x + b_g$ 가 있다고 가정합니다.
> 
> **1) 두 선형 함수의 단순 합:**
> 
> $$(f+g)(x) = (W_f x + b_f) + (W_g x + b_g) = (W_f + W_g)x + (b_f + b_g)$$
> 
> **2) 두 선형 함수의 합성 $h(x) = f(g(x))$:**
> 
> $$h(x) = W_f(W_g x + b_g) + b_f = (W_f W_g)x + (W_f b_g + b_f)$$
> 
> **3) 결론:**
> 
> 괄호를 풀고 정리해 보면 결국 새로운 가중치 $W_{new} = W_f W_g$ 와 새로운 편향 $b_{new} = W_f b_g + b_f$ 를 가지는 **또 다른 하나의 선형 함수**가 될 뿐입니다.
> 
> 따라서 은닉층을 아무리 깊게 쌓아도 활성화 함수가 선형이면 단층 선형 모델과 차이가 없으므로, **꺾이는 곡선 경계선을 형성하기 위해서는 비선형 활성화 함수가 필수적**입니다.

#### 주요 활성화 함수 종류와 수식 및 미분 공식

##### ① Sigmoid (시그모이드)

출력값을 $0 \sim 1$ 사이로 압축하여 확률값으로 해석하기 용이하지만, 깊은 신경망에서는 기울기 소멸(Gradient Vanishing) 문제가 발생합니다.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

> [!tip] Sigmoid 미분 공식 유도
> 
> $$\sigma'(z) = \frac{d}{dz}(1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$
> 
> $$\sigma'(z) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \frac{1}{1 + e^{-z}} \left(1 - \frac{1}{1 + e^{-z}}\right) = \sigma(z)(1 - \sigma(z))$$

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

##### ② Tanh (하이퍼볼릭 탄젠트)

출력 범위를 $-1 \sim 1$로 변환하여 출력값의 평균을 0으로 맞춥니다(Zero-centered). 시그모이드보다 경사하강법 수렴 속도가 빠르지만 여전히 기울기 소멸 문제가 존재합니다.

$$\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

$$\sigma'(z) = 1 - \sigma(z)^2$$

##### ③ ReLU (Rectified Linear Unit)

양수 입력에 대해 기울기가 1로 유지되어 역전파 시 기울기 소멸 문제를 완화하고 연산 속도가 매우 빠릅니다.

$$\text{ReLU}(z) = \begin{cases} z, & z > 0 \\ 0, & \text{otherwise} \end{cases}$$

$$f'(z) = \begin{cases} 1, & z \ge 0 \\ 0, & z < 0 \end{cases}$$

##### ④ LeakyReLU

ReLU에서 $z < 0$일 때 노드가 완전히 죽는 Dying ReLU 현상을 방지하기 위해 음수 영역에 작은 경사값($\alpha$)을 부여합니다.

$$\text{LeakyReLU}(z) = \begin{cases} z, & z > 0 \\ \alpha z, & \text{otherwise} \end{cases} \quad (\text{예: } \alpha=0.2)$$

$$f'(z) = \begin{cases} 1, & z > 0 \\ \alpha, & z \le 0 \end{cases}$$

##### PyTorch 코드로 보는 활성화 함수 연산 예시


```
import torch
import torch.nn as nn
import torch.nn.functional as F

# 샘플 텐서 생성
z = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0], requires_grad=True)

# 1. Sigmoid 및 미분
sigmoid_out = torch.sigmoid(z)
sigmoid_grad = sigmoid_out * (1 - sigmoid_out)

# 2. ReLU 및 LeakyReLU
relu_out = F.relu(z)
leaky_relu_out = F.leaky_relu(z, negative_slope=0.2)

print("=== Activation Outputs ===")
print("Input Z        :", z.detach().numpy())
print("Sigmoid        :", sigmoid_out.detach().numpy())
print("ReLU           :", relu_out.numpy())
print("LeakyReLU(0.2) :", leaky_relu_out.numpy())
```

#### 공식 문서 및 참고 링크

- [PyTorch Official Documentation - torch.nn.functional Activation Functions](https://www.google.com/search?q=https://pytorch.org/docs/stable/nn.functional.html%23non-linear-activation-functions)
    
- [Deep Learning Book (Goodfellow et al.) - Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
    
- [CS231n - Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/)