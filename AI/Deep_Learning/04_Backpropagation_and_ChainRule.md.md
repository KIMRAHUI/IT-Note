---
date: 2026-07-29
tag: [DeepLearning, Backpropagation, ChainRule, GradientDescent, Insight]
status: complete
---

#### 개요

인공신경망 학습의 핵심 알고리즘인 오차 역전파(Backpropagation)의 정의와 작동 원리를 정리합니다. 신경망의 출력층에서 발생한 오차를 역방향으로 전달하면서 각 가중치(Weight)의 기울기(Gradient)를 연쇄법칙(**Chain Rule**)을 통해 유도하는 수학적 과정을 단계별로 상세히 다룹니다.

#### 1. 오차 역전파 (Backpropagation)의 정의와 풀이

- **정의:** 순전파(Forward Propagation)를 통해 얻은 예측값과 실제 정답 사이의 손실(Loss/Cost)을 출력층에서부터 입력층 방향으로 **역방향(Backwards)** 전파하여, 각 층의 가중치($W$)와 편향($b$)에 대한 손실 함수의 편미분값(Gradient)을 산출하고 경사하강법으로 매개변수를 업데이트하는 기법입니다.
    
- **핵심 원리:** 출력층과 가까운 층의 가중치부터 시작하여 연쇄법칙(Chain Rule)을 적용해 은닉층과 입력층 방향으로 미분값을 연쇄적으로 곱해나갑니다.
    

#### 체인룰 (Chain-Rule) 수학적 유도 및 풀이 과정

##### ① 합성함수 미분법 (Chain Rule) 공식

합성함수 $f(g(x))$를 $x$에 대해 미분할 때, 겉미분과 속미분의 곱으로 나타냅니다.

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$$

> [!note] $(x^2 + 1)^2$ 미분 예시 풀이 과정
> 
> 1. **합성함수 치환:** $y = u^2$, $\quad u = x^2 + 1$
>     
> 2. **겉미분 계산:** $\frac{dy}{du} = 2u$
>     
> 3. **속미분 계산:** $\frac{du}{dx} = 2x$
>     
> 4. **체인룰 적용 연쇄 곱:**
>     
>     $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 2x = 2(x^2 + 1)(2x) = 4x^3 + 4x$$
>     

##### ② 첫 번째 은닉층 가중치($w_{11}^{(1)}$)에 대한 오차($E$)의 기울기 최종 유도 풀이

2층 인공신경망에서 입력 $x_1$, 은닉층 출력 $a_{h1}$, 출력층 예측값 $y$, 실제 정답 $y_t$가 존재할 때, 첫 번째 은닉층 가중치 $w_{11}^{(1)}$에 대한 손실 함수 $E$의 미분값 유도 과정입니다.

$$\frac{\partial E}{\partial w_{11}^{(1)}} = (y - y_t) \cdot y(1-y) \cdot w_{13}^{(2)} \cdot a_{h1}(1-a_{h1}) \cdot x_1$$

> [!tip] 단계별 체인룰 유도 과정
> 
> 손실 함수를 오차 제곱합 $E = \frac{1}{2}(y - y_t)^2$ 이라 할 때:
> 
> 1. **출력층 오차 미분 ($\frac{\partial E}{\partial y}$):**
>     
>     $$\frac{\partial E}{\partial y} = y - y_t$$
>     
> 2. **출력층 활성화 함수(Sigmoid) 미분 ($\frac{\partial y}{\partial z_{out}}$):**
>     
>     $$\frac{\partial y}{\partial z_{out}} = y(1 - y)$$
>     
> 3. **출력층 가중합 미분에 따른 은닉층 전달값 ($\frac{\partial z_{out}}{\partial a_{h1}}$):**
>     
>     $$\frac{\partial z_{out}}{\partial a_{h1}} = w_{13}^{(2)}$$
>     
> 4. **은닉층 활성화 함수(Sigmoid) 미분 ($\frac{\partial a_{h1}}{\partial z_{h1}}$):**
>     
>     $$\frac{\partial a_{h1}}{\partial z_{h1}} = a_{h1}(1 - a_{h1})$$
>     
> 5. **입력 가중합 미분에 따른 입력값 전달 ($\frac{\partial z_{h1}}{\partial w_{11}^{(1)}}$):**
>     
>     $$\frac{\partial z_{h1}}{\partial w_{11}^{(1)}} = x_1$$
>     
> 
> **최종 체인룰 합성 연쇄 곱:**
> 
> $$\frac{\partial E}{\partial w_{11}^{(1)}} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial z_{out}} \cdot \frac{\partial z_{out}}{\partial a_{h1}} \cdot \frac{\partial a_{h1}}{\partial z_{h1}} \cdot \frac{\partial z_{h1}}{\partial w_{11}^{(1)}}$$
> 
> $$= (y - y_t) \cdot y(1-y) \cdot w_{13}^{(2)} \cdot a_{h1}(1-a_{h1}) \cdot x_1$$

##### PyTorch Autograd를 활용한 자동 미분 및 기울기 연산 예시


```
import torch

# 1. 입력값 및 정답 텐서 정의 (기울기 추적 활성화)
x1 = torch.tensor([1.0], requires_grad=True)
y_t = torch.tensor([0.0])

# 2. 가중치 파라미터 초기화
w11_1 = torch.tensor([0.5], requires_grad=True) # 첫 번째 은닉층 가중치
w13_2 = torch.tensor([0.8], requires_grad=True) # 출력층 가중치

# 3. 순전파 (Forward Pass)
z_h1 = x1 * w11_1
a_h1 = torch.sigmoid(z_h1)

z_out = a_h1 * w13_2
y = torch.sigmoid(z_out)

# 4. 손실 함수 (MSE)
loss = 0.5 * (y - y_t) ** 2

# 5. 역전파 수행 (Backpropagation)
loss.backward()

print("=== PyTorch Autograd Gradient 계산 결과 ===")
print("Loss (오차 값)                :", loss.item())
print("dE / dw11_1 (w11_1에 대한 기울기):", w11_1.grad.item())
```

#### 공식 문서 및 참고 링크

-  [PyTorch Official Documentation - Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
    
-  [Calculus on Computational Graphs: Backpropagation (Colah's Blog)](https://colah.github.io/posts/2015-08-Backprop/)
    
- [Understanding the Gradient Flow through Batch Normalization Layer (Kratzert)](https://www.google.com/search?q=https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-batch-normalization-layer.html)