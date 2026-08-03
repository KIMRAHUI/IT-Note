---
tags:
  - Deep_Learning
  - AutoEncoder
  - PyTorch
created: 2026-07-31
---

#### 개요

오토인코더의 아키텍처 수식 스펙과 손실 함수(Reconstruction Loss)의 수학적 정의를 다룹니다. PyTorch 기반의 **Dense Layer 오토인코더** 및 Convolutional AutoEncoder (CAE)의 전체 실습 코드와 계층별 특징 추출 메커니즘을 상세히 비교합니다. 또한 잡음 제거 오토인코더 (Denoising AutoEncoder, DAE)와 **이미지 생성 (Image Generation)** 응용 기법을 종합 정리합니다.

#### AutoEncoder 아키텍처 및 손실 함수

##### ① 오토인코더의 수학적 구조 스펙 (레몬 이미지 예시)

```
[Input 480x360] ──> [Hidden1 1000] ──> [Hidden2 500] ──> [Latent z 10D] ──> [Hidden3 500] ──> [Hidden4 1000] ──> [Output 480x360]
```

- **학습 성격:** 오토인코더는 입력값 $x$ 자체를 목표 출력 $x'$로 재구성하므로, 픽셀 단위 연속값을 예측하는 **비지도 회귀(Regression) 문제**로 접근합니다.
    

##### ② 재구성 손실 함수 (Reconstruction Loss)

$$\mathcal{L}(x, x') = \Vert x - x' \Vert^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - x_i')^2$$

- **MSE (Mean Squared Error):** 원본 픽셀 값 $x$와 복원 픽셀 값 $x'$ 간의 평균 제곱 오차.
    
- **BCE (Binary Cross Entropy):** 픽셀 값이 $0 \sim 1$로 정규화된 흑백 이미지(MNIST 등)에서 손실 함수로 대체 사용 가능.
    

#### PyTorch 실습: Dense vs Convolutional AutoEncoder

##### ① Dense Layer 기반 오토인코더 (PyTorch 전체 코드)

```
import torch
import torch.nn as nn

class DenseAutoencoder(nn.Module):
    def __init__(self):
        super(DenseAutoencoder, self).__init__()
        # Encoder: 784 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # Decoder: 32 -> 64 -> 128 -> 784 (인코더와 대칭 구조)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = DenseAutoencoder()
print(model)
```

##### ② Convolutional AutoEncoder (CAE) (PyTorch 코드)

```
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder: Conv2d + MaxPool로 공간 구조 보존하며 압축
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (Batch, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (Batch, 32, 7, 7)
        )
        # Decoder: ConvTranspose2d (또는 Upsample + Conv2d) 구조
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # (Batch, 16, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # (Batch, 1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

##### ③ Dense 기반 AE vs CNN 기반 AE 비교

| **비교 항목**     | **Dense 기반 오토인코더**           | **CNN 기반 오토인코더 (CAE)**                                                       |
| ------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| **입력 데이터 처리** | 이미지를 1차원 벡터로 Flatten 변환      | 이미지의 2D 공간 구조(Spatial Structure)를 그대로 유지                                     |
| **계층적 특징 학습** | 저차원 단순 수치 결합 위주 학습           | Low-level (엣지/색상) $\rightarrow$ Mid-level (질감) $\rightarrow$ High-level (객체) |
| **복원 이미지 품질** | 세밀한 공간 정보 손실로 복원물이 상대적으로 흐릿함 | 공간 정보 보존으로 복원물이 원본에 가깝게 선명함                                                  |

#### 3. 잡음 제거 오토인코더 (Denoising AutoEncoder, DAE)

##### ① 학습 메커니즘 및 수식

Denoising AutoEncoder는 의도적으로 입력 데이터에 노이즈를 추가한 후, **노이즈가 제거된 깨끗한 원본 데이터로 복원**하도록 학습시키는 변형 구조입니다.

- **입력 데이터 (**$p$**):** $p = x + r$ ($x$: 깨끗한 원본 데이터, $r$: 무작위 가우시안/솔트페퍼 잡음)
    
- **목표 출력 (Target):** 노이즈가 없는 깨끗한 원본 $x$  
    
- **손실 함수:** `L_DAE = || x - f_theta(g_phi(p)) ||^2`
    

```
[노이즈 입력 p = x + r] ──> [Encoder] ──> [Bottleneck z] ──> [Decoder] ──> [깨끗한 원본 복원 x']
```

##### ② Denoising AE의 핵심 이점

1. **강건한 특징 (Robust Feature) 추출:** 노이즈와 같은 무의미한 요소를 병목 구간에서 자연스럽게 차단하여, 데이터의 강건한 본질적 구조만 학습합니다.
    
2. **복원 및 잡음 제거 (Image Denoising):** 실제 노이즈가 포함된 훼손된 데이터가 입력되어도 원본 데이터를 선명하게 복원해냅니다.
    

#### 오토인코더 주요 응용 형태 종합 비교

| **분류**                        | **입력 (X)**             | **목표 출력 (Target)**   | **핵심 목적 및 활용 분야**         |
| ----------------------------- | ---------------------- | -------------------- | ------------------------- |
| **기본 오토인코더 (Vanilla AE)**     | 원본 데이터 $x$             | 원본 데이터 $x$           | 차원 축소, 시각화, 특징 추출         |
| **잡음 제거 오토인코더 (DAE)**         | 노이즈 포함 데이터 $p = x + r$ | **깨끗한 원본** $x$       | 이미지 잡음 제거, 강건한 특징 학습      |
| **이미지 생성 (Image Generation)** | 원본 $x$ 또는 샘플링 $z$      | **변형된 / 새 이미지** $x'$ | 잠재 공간 조작/보간을 통한 신규 이미지 합성 |

#### 공식 문서 및 참고 링크

-  [PyTorch Official Documentation - torch.nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html "null")
    
-  [Journal of Machine Learning Research - Extracting and Composing Robust Features with Denoising Autoencoders (Vincent et al., 2010)](https://www.jmlr.org/papers/v11/vincent10a.html "null")
    
-  [PyTorch Official Tutorials - Adversarial Autoencoders and Generative Models](https://pytorch.org/tutorials/ "null")
