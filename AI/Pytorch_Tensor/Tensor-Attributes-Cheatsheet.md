---
tags:
  - deep-learning
  - pytorch
  - tensor
  - dtype
  - DataModeling
created: 2026-08-04
---

---

<div style="background-color: rgba(100, 150, 250, 0.1); padding: 12px 16px; border-radius: 6px; border-left: 4px solid #4a90e2;">
  <b>텐서 디버깅 5가지</b><br>
  에러가 나거나 상태를 확인할 때 가장 먼저 <code>print()</code>로 찍어봐야 할 5가지 속성입니다.<br><br>
  1. <code>tensor.shape</code> (크기)<br>
  2. <code>tensor.dtype</code> (데이터 타입)<br>
  3. <code>tensor.device</code> (장치 위치: CPU/GPU)<br>
  4. <code>tensor.requires_grad</code> (역전파 여부)<br>
  5. <code>tensor.ndim</code> (차원 수)
</div>

```python
import torch

t = torch.rand(2, 3, requires_grad=True)
print("shape:", t.shape)
print("size():", t.size())
print("ndim:", t.ndim)
print("dtype:", t.dtype)
print("device:", t.device)
print("requires_grad:", t.requires_grad)
```

**출력 예시:**

```
shape: torch.Size([2, 3])
size(): torch.Size([2, 3])
ndim: 2
dtype: torch.float32
device: cpu
requires_grad: True
```

##### 속성별 상세 설명

|분류|속성 / 메서드|설명 및 역할|
|---|---|---|
|기본 정보|`.shape` / `.size()`|텐서의 전체 형태(크기)를 반환합니다. (예: `torch.Size([2, 3])`)|
|차원 수|`.ndim`|텐서의 차원 수(Dimensions)를 정수로 반환합니다.|
|데이터 타입|`.dtype`|텐서에 저장된 데이터 타입을 확인합니다. (예: `torch.float32`, `torch.int64`)|
|연산 장치|`.device`|텐서가 할당된 장치(CPU 또는 GPU)를 확인합니다. (예: `cpu`, `cuda:0`)|
|미분 추적|`.requires_grad`|자동 미분(Autograd) 기록 여부를 불리언(`True`/`False`)으로 반환합니다.|

#####  `.shape` / `.size()` — 형태 확인

- `.shape`는 속성(property), `.size()`는 메서드(method)라서 괄호 유무만 다르고 반환값(`torch.Size` 객체)은 완전히 같습니다.
- `torch.Size`는 사실상 튜플(tuple)이라서 `t.shape[0]`처럼 인덱싱해서 특정 축의 크기만 뽑아낼 수도 있습니다.

```python
print(t.shape[0])  # 2 (행의 개수)
print(t.shape[1])  # 3 (열의 개수)
```

#####  `.ndim` — 차원 수

- 스칼라(0차원), 벡터(1차원), 행렬(2차원), 그 이상(3차원 이상)을 구분할 때 씁니다.

```python
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.rand(2, 3)

print(scalar.ndim)  # 0
print(vector.ndim)  # 1
print(matrix.ndim)  # 2
```

#####  `.dtype` — 데이터 타입

- 기본적으로 `torch.rand()`, `torch.randn()` 등으로 만든 텐서는 `torch.float32`가 기본값입니다.
- 정수 텐서는 보통 `torch.int64`(=`torch.long`)가 기본값입니다.
- 딥러닝 모델 학습 시 두 텐서의 `dtype`이 다르면 연산 중 에러가 나므로 항상 확인하는 습관이 중요합니다. (자세한 내용은 [Dtype-Conversion-NumPy-Interop](./Dtype-Conversion-NumPy-Interop.md) 참고)

#####  `.device` — 연산 장치

- 텐서가 `cpu`에 있는지 `cuda:0`(GPU 0번)에 있는지를 알려줍니다.
- 서로 다른 device에 있는 텐서끼리 연산하면 에러가 발생합니다. (자세한 내용은 [Device-Management-CPU-GPU](./Device-Management-CPU-GPU.md) 참고)

##### `.requires_grad` — 자동 미분 추적 여부

- `True`로 설정하면 이 텐서에 대해 이루어지는 모든 연산이 기록되어, 나중에 `.backward()`를 호출했을 때 기울기(gradient)를 자동으로 계산할 수 있습니다.
- 주로 모델의 학습 가능한 파라미터(가중치, 편향)에 `requires_grad=True`를 설정합니다. 입력 데이터나 정답 라벨은 보통 `False`(기본값)로 둡니다.

```python
w = torch.randn(3, requires_grad=True)
y = (w ** 2).sum()
y.backward()
print(w.grad)  # w에 대한 미분값(기울기)이 출력됨
```

#### 자주 하는 실수

- `.shape`를 함수처럼 `t.shape()`로 호출하면 에러가 납니다. `.shape`는 속성이므로 괄호 없이 써야 합니다. 괄호가 필요한 것은 `.size()`입니다.
- `requires_grad=True`인 텐서를 바로 `.numpy()`로 변환하려 하면 에러가 납니다. `.detach().numpy()`처럼 grad 추적을 끊어줘야 합니다.

---

#### 관련 노트

- [PyTorch-Tensor-Data-Modeling-MOC](./PyTorch-Tensor-Data-Modeling-MOC.md)
- [Tensor-Shape-Interpretation](./Tensor-Shape-Interpretation.md)
- [Device-Management-CPU-GPU](./Device-Management-CPU-GPU.md)
- [Dtype-Conversion-NumPy-Interop](./Dtype-Conversion-NumPy-Interop.md)
