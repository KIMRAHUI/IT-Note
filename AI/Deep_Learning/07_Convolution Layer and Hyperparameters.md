---
tags:
  - Machine_Learning
  - Deep_Learning
  - CNN
created: 2026-07-30
---

#### 개요

CNN의 핵심 연산을 담당하는 합성곱 층(Convolutional Layer)의 수학적 동작 원리와 주요 하이퍼파라미터(**Kernel Size, Stride, Padding**)를 상세히 다룹니다. 출력 특징 맵(Feature Map)의 크기 및 학습 파라미터 수 계산 공식을 풀이하고, 합성곱 연산의 선형성을 극복하기 위한 비선형 활성화 함수(ReLU)의 필요성을 다룹니다.

#### Convolution (합성곱) 연산과 수학적 수식

합성곱 연산은 입력 데이터와 필터(커널) 간의 **원소별 곱의 합(Summation of Element-Wise Multiplication / Dot Product)** 및 **편향(Bias)** 추가로 이루어집니다.

$$y(i, j) = \sum_{m} \sum_{n} \sum_{c} x(i+m, j+n, c) \cdot k(m, n, c) + b$$

- $x$: 입력 데이터 텐서 (위치 $i+m, j+n$, 채널 $c$)
    
- $k$: 커널(필터) 가중치 텐서 (위치 $m, n$, 채널 $c$)
    
- $b$: 편향 (Bias, 필터당 1개)
    
- $y(i, j)$: 출력 특징 맵(Feature Map)의 특정 위치 수치
    

#### 주요 하이퍼파라미터 비교

| **하이퍼파라미터**                      | **정 의**                      | **설정 값 및 특성**                                                       |
| -------------------------------- | ---------------------------- | ------------------------------------------------------------------- |
| **커널 크기 (Kernel Size,** $K$**)** | 필터의 가로 $\times$ 세로 행렬 크기     | 주로 $3 \times 3$, $5 \times 5$, $7 \times 7$ 등 **홀수** 사용 (중심점 존재 목적) |
| **스트라이드 (Stride,** $S$**)**      | 필터를 이미지상에서 이동시키는 간격(칸 수)     | Stride가 커질수록 출력 크기는 작아짐 (보통 1 또는 2 설정)                              |
| **패딩 (Padding,** $P$**)**        | 입력 가장자리에 $0$ 등의 값을 덧대어 크기 확장 | 외곽 픽셀의 연산 횟수 부족으로 인한 정보 소실 방지 및 크기 유지                               |

#### 패딩(Padding)의 3가지 유형 비교

| **패딩 종류**         | **방식 및 개념**                | **출력 크기 변화**         | **주요 사용처**            |
| ----------------- | -------------------------- | -------------------- | --------------------- |
| **Valid Padding** | 패딩을 전혀 적용하지 않음<br> ($P=0$) | 입력보다 출력 크기가 줄어듦      | 연산량 축소, 단순 특징 추출 시    |
| **Same Padding**  | 입력 크기와 출력 크기가 동일하도록 패딩 추가  | **입력 크기 = 출력 크기 유지** | **CNN 모델에서 가장 널리 사용** |
| **Full Padding**  | 필터의 끝부분만 걸쳐도 연산되도록 넓게 패딩   | 입력보다 출력 크기가 커짐       | 특수한 확장 컨볼루션 연산 시      |

#### 특징 맵 크기 및 파라미터 수 계산 풀이

##### ① 출력 특징 맵 (Feature Map) 크기 계산 공식

$$\text{Output Size} = \left\lfloor \frac{H - K + 2P}{S} \right\rfloor + 1$$

- $H$: 입력 높이/너비 (Square 기준)
    
- $K$: 커널 크기, $P$: 패딩 크기, $S$: 스트라이드 크기
    
- $\lfloor \dots \rfloor$: 버림(Floor) 연산. 격자 연산 특성상 불완전하게 걸치는 마지막 위치는 버림 처리됩니다.
    

> [!note] 계산 예제 풀이
> 
> **예제 1:** 입력 $128 \times 128 \times 30$, 커널 $5 \times 5$, Stride = 2, Padding = 2
> 
>   
> 
> $$\text{Output} = \left\lfloor \frac{128 - 5 + 2(2)}{2} \right\rfloor + 1 = \left\lfloor \frac{127}{2} \right\rfloor + 1 = 63 + 1 = 64 \quad \longrightarrow \mathbf{64 \times 64}$$
> 
> **예제 2:** 입력 $64 \times 64 \times 16$, 커널 $3 \times 3$, 필터 32개, Stride = 1, Valid Padding ($P=0$)
> 
>   
> 
> $$\text{Output} = \left\lfloor \frac{64 - 3 + 0}{1} \right\rfloor + 1 = 61 + 1 = 62 \quad \longrightarrow \mathbf{62 \times 62 \times 32}$$

##### ② 채널 법칙 및 학습 파라미터 수 계산 공식

|   |   |
|---|---|
|**채널 법칙**|**상세 설명**|
|**기칙 1**|**커널 1개의 채널 수(깊이) = 입력 데이터의 채널 수 (**$C_{in}$**)**|
|**기칙 2**|**출력 특징 맵의 채널 수 (**$C_{out}$**) = 사용한 커널(필터)의 개수 (**$N$**)**|
|**기칙 3**|채널별 원소 곱 후 전체 합을 수행하여 단 1개의 2D 특징 맵 출력|

$$\text{Total Params} = \Big( (K_H \times K_W \times C_{in}) + 1 \Big) \times C_{out}$$

> [!note] 파라미터 계산 예시
> 
> 입력이 $224 \times 224 \times 3$ 이고, $3 \times 3$ 커널을 가진 필터 2개를 적용하는 경우:
> 
> - 필터 1개당 가중치: $3 \times 3 \times 3 = 27$개
>     
> - 필터 1개당 편향: $1$개
>     
> - 필터 2개의 총 파라미터 수: $(27 + 1) \times 2 = 56$개
>     

#### 선형 연산과 비선형성(ReLU)의 추가

| **연산 구분**                      | **성 격**                                      | **한계점 및 필요성**                          |
| ------------------------------ | -------------------------------------------- | -------------------------------------- |
| **Convolution 연산**             | 가중곱합(Weighted Sum)으로 이루어진 **선형 연산 (Linear)** | 활성화 함수 없이 Conv만 쌓으면 결국 1개의 선형 변환으로 축소됨 |
| **Activation Function (ReLU)** | 입력값에 비선형성을 부여하는 **비선형 연산 (Non-linear)**      | 복잡한 곡선 형태의 결정 경계 및 고수준 비선형 패턴 학습 가능    |

##### PyTorch 코드로 보는 Conv2d 연산 및 Output Shape 계산 예시

```
import torch
import torch.nn as nn

# 1. Conv2d 레이어 정의 (Input Channel=3, Output Channel=32, Kernel=3, Stride=1, Padding=1)
conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

# 2. 샘플 입력 텐서 생성 (Batch=4, Channel=3, Height=64, Width=64)
x = torch.randn(4, 3, 64, 64)
output = conv_layer(x)

# 3. 출력 크기 및 파라미터 수 확인
print("=== Conv2d Output Shape ===")
print("Output Shape:", output.shape) # [4, 32, 64, 64] (Same Padding 적용)

# 파라미터 계산: ((3 * 3 * 3) + 1) * 32 = 896
params_count = sum(p.numel() for p in conv_layer.parameters())
print(f"Total Conv Parameters: {params_count} 개")
```

#### 공식 문서 및 참고 링크

-  [PyTorch Official Documentation - torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html "null")
    
-  [Towards AI - An Introduction to CNNs: Understanding the Basics](https://pub.towardsai.net/an-introduction-to-cnns-understanding-the-basics-88986fb3c6d1 "null")
    
-  [Deep Learning Book (Goodfellow et al.) - Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html "null")