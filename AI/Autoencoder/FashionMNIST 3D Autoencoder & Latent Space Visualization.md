---
tags:
  - pytorch
  - AutoEncoder
  - LatentSpace
  - Plotly
  - Scatter3d
  - torchvision
  - view
  - FashionMNIST
created: 2026-08-06
---
----

#### 개요 및 환경 설정

**실습 목적** 고차원의 의류 이미지 데이터를 오토인코더(Autoencoder)의 모래시계 구조를 통해 압축하고, 3차원 잠재 공간(Latent Space)으로 축소한 뒤 이를 Plotly 기반의 인터랙티브 3D 공간에 시각화합니다.

**사용 데이터셋**

- **FashionMNIST** — 학습용 60,000장 / 테스트용 10,000장의 흑백 의류 이미지(28×28)로 구성된 데이터셋입니다. 

**전처리 파이프라인 (`v2.Compose`)**

- `v2.ToImage()`: 이미지를 PyTorch 텐서 형식으로 변환
- `v2.ToDtype(dtype=torch.float32, scale=True)`: 픽셀 값의 타입을 `float32`로 바꾸고, `[0, 255]` 범위를 `[0.0, 1.0]` 사이의 실수로 자동 정규화(Scaling)

```python
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
```

> ⚠️  놓치기 쉬운 포인트
> 
> - `v2`는 `torchvision.transforms`의 **개선된 버전**입니다. 예전 코드에서 자주 보이는 `transforms.ToTensor()` 하나로 끝내던 방식 대신, 최신 방식은 `v2.ToImage()` + `v2.ToDtype(..., scale=True)`를 조합해서 씁니다. 두 줄 다 있어야 "텐서 변환"과 "0~1 정규화"가 모두 처리됩니다.
> - `scale=True`를 빼먹으면 픽셀 값이 0~255 정수 범위 그대로 모델에 들어가서, 학습이 극도로 불안정해지거나 loss가 처음부터 매우 크게 나옵니다.

---

#### 오토인코더 아키텍처 (`Autoencoder3D`)

고차원 입력을 병목 구간(Bottleneck)으로 좁혔다가 다시 늘려주는 모래시계 형태의 대칭 구조입니다.

**인코더 (Encoder — 압축 단계)**

- 구조: `Linear(784→128)` → `ReLU` → `Linear(128→64)` → `ReLU` → `Linear(64→12)` → `ReLU` → `Linear(12→3)`
- 역할: 28×28 이미지(784차원)를 핵심 특징만 남긴 채 3차원(Latent Space)으로 꾹 압축

**디코더 (Decoder — 복원 단계)**

- 구조: `Linear(3→12)` → `ReLU` → `Linear(12→64)` → `ReLU` → `Linear(64→128)` → `ReLU` → `Linear(128→784)` → `Sigmoid`
- 역할: 3차원 잠재 벡터를 받아 다시 원래 크기인 784차원으로 부풀리고, `Sigmoid`를 통해 픽셀 값 범위(`[0, 1]`)를 맞춰 원본 이미지 복원

**핵심 추가 메서드 (`encode`)**

- 모델 전체를 거치지 않고 입력 데이터의 3차원 잠재 벡터만 별도로 추출할 수 있도록 구현된 메서드 (`return self.encoder(x)`)

```python
import torch
import torch.nn as nn

class Autoencoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 12), nn.ReLU(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), nn.ReLU(),
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)
```

> ⚠️ 놓치기 쉬운 포인트
> 
> - 오토인코더는 **입력과 정답(타깃)이 사실상 같은 이미지**입니다. 즉 "이미지를 넣고 → 압축했다가 → 최대한 똑같이 복원"하는 것이 목표라서, 손실 함수 계산 시 `loss_fn(reconstructed, original_image)`처럼 타깃 자리에 원본 이미지 자체를 넣습니다. 이 부분이 지도학습(분류/회귀)과 가장 다른 점입니다.
> - 마지막 디코더 출력에 `Sigmoid`를 쓰는 이유는, 입력 이미지를 `[0, 1]` 범위로 정규화했기 때문에 복원값도 같은 범위로 나오게 강제하기 위해서입니다. 만약 입력을 다른 범위로 정규화했다면 마지막 활성화 함수도 그에 맞게 바꿔야 합니다.
> - `encode()`처럼 `forward()` 외에 별도의 메서드를 자유롭게 추가할 수 있다는 걸 기억하세요. `nn.Module`을 상속받은 클래스는 일반 파이썬 클래스이므로 원하는 메서드를 얼마든지 더 만들 수 있습니다. 단, `model(x)`처럼 객체를 직접 호출하면 항상 `forward()`가 실행되고, `encode()`처럼 이름이 다른 메서드는 `model.encode(x)`로 명시적으로 호출해야 합니다.

---

#### 학습 및 최적화 루프 (Optimization Loop)

모든 딥러닝 모델 학습은 아래 **5단계의 순환 루프**를 통해 이루어집니다.

1. **Forward (순전파):** `outputs = model(images)` → 입력 이미지를 모델에 통과시켜 복원 이미지 생성
2. **Loss 계산 (오차 측정):** `loss = loss_fn(outputs, images)` → 원본 이미지와 복원된 이미지 간의 평균 제곱 오차(`MSELoss`) 측정
3. **Zero_grad (기울기 초기화):** `optimizer.zero_grad()` → 이전 스텝에서 누적된 기울기(Gradient) 버퍼를 리셋하여 중첩 방지
4. **Backward (역전파):** `loss.backward()` → 오차를 바탕으로 각 파라미터의 미분값(기울기) 계산
5. **Step (가중치 갱신):** `optimizer.step()` → `Adam` 옵티마이저가 학습률(`lr=0.0001`)을 적용해 모델 가중치 업데이트

```python
import torch.optim as optim

model = Autoencoder3D().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 20
for epoch in range(epochs):
    model.train()
    for images, _ in train_dataloader:
        images = images.view(images.size(0), -1).to(device)  # (batch, 1, 28, 28) → (batch, 784)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, images)   # 타깃이 원본 이미지 자기 자신
        loss.backward()
        optimizer.step()

    print(f'epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}')
```

> ⚠️ 놓치기 쉬운 포인트
> 
> - `images.view(images.size(0), -1)`는 `(batch, 1, 28, 28)` 형태의 4차원 이미지 텐서를 `(batch, 784)` 형태의 2차원 텐서로 **평탄화(flatten)** 하는 코드입니다. `nn.Linear`는 마지막 축이 `in_features`와 맞는 2차원(또는 그 이상) 텐서를 기대하므로, 이미지처럼 다차원인 데이터는 반드시 이렇게 펼쳐줘야 합니다. `-1`은 "나머지 크기는 알아서 계산해줘"라는 뜻입니다.
> - 여기서 쓰는 옵티마이저는 앞서 다룬 `SGD`가 아니라 **`Adam`** 입니다. `Adam`은 학습률을 파라미터별로 자동 조정해주는 알고리즘이라 대체로 `SGD`보다 수렴이 빠르고 안정적이어서, 실무에서 기본으로 가장 많이 쓰입니다.
> - 라벨(`_`)을 `for images, _ in train_dataloader:`처럼 언더스코어로 받아서 버리고 있는 이유는, 오토인코더 학습 자체에는 클래스 라벨이 전혀 필요 없기 때문입니다(비지도학습). 라벨은 나중에 잠재 공간을 시각화할 때 색상 구분용으로만 사용됩니다.

---

#### 3차원 잠재 공간(Latent Space) 추출 코드

```python
# 1. 모델을 평가 모드(Evaluation Mode)로 전환 (Dropout/BatchNorm 고정)
model.eval()

all_latents = []
all_labels = []

# 2. 평가 단계이므로 역전파 연산 그래프 저장을 비활성화하여 메모리 절약
with torch.no_grad():
    for images, labels in test_dataloader:
        # 1) 4차원 텐서([Batch, 채널, 세로, 가로])를 2차원 평탄화([Batch, 784])한 뒤 장치로 이동
        images = images.view(images.size(0), -1).to(device)

        # 2) 전체 모델이 아닌 인코더만 거쳐 3차원 잠재 벡터 추출
        latent = model.encode(images)

        # 3) CPU로 옮긴 뒤 NumPy 배열로 변환하여 리스트에 축적
        all_latents.append(latent.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# 리스트에 담긴 미니배치 배열들을 axis=0 방향으로 결합하여 전체 테스트 셋 완성
all_latents = np.concatenate(all_latents, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
```

> ⚠️ 놓치기 쉬운 포인트
> 
> - `model.eval()` + `torch.no_grad()`를 함께 쓰는 이유는 이전 노트들에서도 다룬 원칙과 동일합니다. 여기서는 학습이 아니라 "결과 추출"만 하는 단계이므로 그래디언트 추적이 전혀 필요 없습니다.
> - `latent.cpu().numpy()`처럼 **GPU → CPU → NumPy** 순서를 항상 지켜야 합니다. GPU 텐서에서 바로 `.numpy()`를 호출하면 에러가 납니다.
> - `np.concatenate(all_latents, axis=0)`은 배치 단위로 쌓인 리스트(`[(32,3), (32,3), ...]`)를 하나의 큰 배열(`(10000, 3)`)로 합치는 역할입니다. `axis=0`은 배치 방향(행 방향)으로 이어붙인다는 뜻입니다. `torch.cat`의 NumPy 버전이라고 생각하면 됩니다.

---

#### Plotly 인터랙티브 3D 시각화

```python
import plotly.graph_objects as go
import numpy as np

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

fig = go.Figure()

for digit in np.unique(all_labels):
    indices = np.where(all_labels == digit)[0]
    # 3D 공간 과부하 방지를 위해 클래스당 최대 50개 샘플 무작위 추출
    sample_indices = np.random.choice(indices, size=min(50, len(indices)), replace=False)

    fig.add_trace(go.Scatter3d(
        x=all_latents[sample_indices, 0],  # Latent Dimension 1
        y=all_latents[sample_indices, 1],  # Latent Dimension 2
        z=all_latents[sample_indices, 2],  # Latent Dimension 3
        mode='text',                       # 점 대신 텍스트 자체를 마커로 사용
        text=[str(digit)] * len(sample_indices),
        textposition='middle center',      # 텍스트 정렬 위치 (중앙)
        textfont=dict(color=colors[int(digit)], size=12),  # 클래스별 고유 색상 및 글자 크기
        name=f"Digit {digit}"              # 범례(Legend) 이름
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='Latent Dimension 1',
        yaxis_title='Latent Dimension 2',
        zaxis_title='Latent Dimension 3'
    ),
    width=800,
    margin=dict(r=20, l=10, b=10, t=10),
    title="3D Visualization of MNIST Latent Space (Text Labels)"
)

fig.show()
```

**🔤 파이썬 문법 핵심 포인트 — 리스트 반복 연산자 `*`**

- `text=[str(digit)] * len(sample_indices)` 부분은 문자열의 길이를 곱하는 것이 아니라, **리스트 안의 문자열을 지정한 개수만큼 반복 생성**하는 문법입니다.
- 예: `['3'] * 5` → `['3', '3', '3', '3', '3']`
- Plotly의 `text` 파라미터는 각 좌표점마다 하나씩 매칭되는 "문자열 리스트"를 요구하기 때문에, 같은 라벨 숫자를 점 개수만큼 반복해서 리스트로 만들어주는 것입니다.

> ⚠️ 놓치기 쉬운 포인트
> 
> - `np.random.choice(indices, size=..., replace=False)`에서 `replace=False`는 "한 번 뽑은 인덱스는 다시 뽑지 않는다"는 뜻(비복원추출)입니다. `replace=True`로 두면 같은 데이터 포인트가 중복으로 뽑힐 수 있습니다.
> - `colors[int(digit)]`처럼 라벨 값을 색상 리스트의 인덱스로 바로 사용하고 있습니다. 이 방식이 동작하려면 라벨이 반드시 `0`부터 시작하는 정수(0, 1, 2, ... 9)여야 합니다. 문자열 라벨이라면 먼저 정수 인덱스로 매핑하는 작업이 필요합니다.
> - `mode='text'`는 일반적인 산점도에서 흔히 쓰는 `mode='markers'`(동그란 점)와 다르게, 점 자리에 **숫자/글자 자체**를 표시합니다. 클래스가 많을 때 점보다 라벨을 직접 보는 게 더 직관적이어서 사용된 방식입니다. 동그란 점으로 표시하고 싶다면 `mode='markers'`로 바꾸고 `marker=dict(color=..., size=...)`를 사용하면 됩니다.



#### 공식 문서 및 참고 링크


- **[FashionMNIST  (Zalando Research)](https://github.com/zalandoresearch/fashion-mnist)** 
- **[torchvision.datasets.FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)** 
- **[torchvision.transforms.v2](https://docs.pytorch.org/vision/stable/transforms.html)** 
- **[torch.optim.Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)** 
- **[torch.nn.MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)** 
- **[Plotly Scatter3d](https://plotly.com/python/3d-scatter-plots/)** 
-


