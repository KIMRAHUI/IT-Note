---
tags:
  - Machine_Learning
  - Deep_Learning
  - CNN
created: 2026-07-30
---


#### 개요

CNN의 수용장(**Receptive Field**) 확장 원리를 파악하고, 특징 맵의 공간적 차원을 줄여 연산량을 감소시키는 Pooling Layer(Max vs Average Pooling)의 특성을 다룹니다. 또한 네트워크 후반부에서 특징 추출 결과를 바탕으로 최종 클래스를 판단하는 Classification Head(Flatten, Fully Connected Layer, Global Average Pooling)의 구조를 정리합니다.

#### Receptive Field (수용장)의 개념 및 확장 원리

- **정의:** CNN에서 특정 층의 뉴런 하나가 입력 이미지의 어느 넓은 범위 <br>(영역)를 참고하여 정보를 받아들이는지를 나타내는 영향 범위입니다.
    

| **네트워크 깊이**             | **뉴런의 수용장 (Receptive Field) 크기** | **학습하는 특징의 수준 (Feature Level)**    |
| ----------------------- | -------------------------------- | ---------------------------------- |
| **초기 층 (Lower Layer)**  | 아주 작고 국소적인 수용장을 가짐               | 선, 에지, 모서리 등 저수준(Low-level) 패턴     |
| **깊은 층 (Deeper Layer)** | 누적 수용장이 점진적으로 넓어짐                | 객체의 부분, 질감, 전역 문맥(Context) 및 전체 형태 |

- **확장 원리:**<br> Stride가 크거나 Pooling 층을 통과할수록 수용장(RF)의 확장 속도가 더욱 빨라집니다.
    

#### Pooling Layer (풀링 층 / 서브샘플링)

특징 맵의 공간적 차원(가로, 세로)을 축소하여 연산량을 줄이고 공간적 불변성을 확보하는 다운샘플링 레이어입니다. **학습되는 가중치가 없으며**, 지정된 연산 방식만 수행합니다.

#### Max Pooling vs Average Pooling 비교

| **구 분**    | **맥스 풀링 (Max Pooling)**       | **에버리지 풀링 (Average Pooling)**          |
| ---------- | ----------------------------- | -------------------------------------- |
| **연산 방식**  | 윈도우 영역 내 최댓값 선택               | 윈도우 영역 내 모든 값의 평균 계산                   |
| **특징 강조**  | 가장 두드러진 특징 강조 (날카로운 에지/패턴 보존) | 영역 전체를 부드럽게 요약 (전반적 통계치 유지)            |
| **신호 보존**  | 강한 특징 신호는 유지, 약한 신호는 제거       | 모든 특징이 반영되나 값이 희석(Blur)될 수 있음          |
| **민감도**    | 특징의 정확한 위치보다 존재 유무에 민감        | 극단값에 둔감하여 출력이 매끄럽고 노이즈 저항 높음           |
| **주요 사용처** | 일반적인 CNN 특징 추출 (분류, 객체 탐지)    | 네트워크 최후반부 Global Average Pooling (GAP) |

- **핵심 특징:** Pooling 연산을 거치면 **가로/세로 크기는 줄어들지만 채널(Depth) 수는 변하지 않고 그대로 유지**됩니다.
    

#### Classification Head 구성 요소 비교

CNN 전체 파이프라인은 <br>크게 Feature Extraction (합성곱 + 풀링)과 Classification (분류 판단)으로 나뉩니다.

| **분류 헤드 구성 요소**                  | **메커니즘 및 역할**                               | **파라미터 수 및 특성**            |
| -------------------------------- | ------------------------------------------- | -------------------------- |
| **Flatten Layer**                | 3차원 특징 맵($H \times W \times C$)을 1차원 벡터로 변환 | 학습 파라미터 0개 (전처리 역할)        |
| **Fully Connected (FC)**         | 추출된 고수준 특징을 전역적으로 결합하여 최종 분류                | 파라미터가 매우 많아 과적합 위험 존재      |
| **Global Average Pooling (GAP)** | 마지막 특징 맵 채널별 전체 영역 평균값으로 축소                 | **학습 파라미터 0개** (과적합 위험 극복) |

#### GAP vs Fully Connected Layer 비교

| **비교 항목**     | **Fully Connected Layer (FC)** | **Global Average Pooling (GAP)** |
| ------------- | ------------------------------ | -------------------------------- |
| **연산 방식**     | Feature Map 전체와 행렬 곱 연산        | 채널별 $(H, W)$ 전체 공간의 평균 연산        |
| **파라미터 수**    | 폭발적으로 증가 (과적합의 주요 원인)          | **0개 (파라미터 없음)**                 |
| **공간 정보 유지**  | Flatten 과정에서 공간 구조 완전 상실       | 채널별 특성을 매핑하여 과적합 완화              |
| **입력 크기 자유도** | 입력 해상도가 바뀌면 전체 재설계 필요          | 입력 해상도가 바뀌어도 $(1, 1, C)$로 축소 가능  |

##### PyTorch 코드로 보는 MaxPool2d 및 GAP 구현 예시

```
import torch
import torch.nn as nn

# 1. 샘플 특징 맵 생성 (Batch=2, Channel=64, Height=8, Width=8)
feature_map = torch.randn(2, 64, 8, 8)

# 2. Max Pooling 적용 (2x2, Stride=2)
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled_out = max_pool(feature_map)
print("=== Max Pooling Output Shape ===")
print("Shape:", pooled_out.shape) # [2, 64, 4, 4]

# 3. Global Average Pooling (GAP) 적용
gap = nn.AdaptiveAvgPool2d((1, 1))
gap_out = gap(feature_map)
print("\n=== Global Average Pooling (GAP) Output Shape ===")
print("Shape:", gap_out.shape) # [2, 64, 1, 1] -> FC Layer 없이 분류 가능
```

#### 공식 문서 및 참고 링크

-  [PyTorch Official Documentation - torch.nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html "null")
    
-  [PyTorch Official Documentation - torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html "null")
    
- [The AI Summer - Understanding Receptive Field in CNNs](https://theaisummer.com/receptive-field/ "null")