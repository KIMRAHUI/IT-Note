---
tags:
  - Deep_Learning
  - rnn
  - bptt
  - vanishing-gradient
  - exploding-gradient
  - gradient-clipping
  - long-term-dependency
created: 2026-08-03
---

#### 개요
본 문서는 순환 신경망(RNN)의 핵심 학습 알고리즘인 BPTT(Backpropagation Through Time, 시간 역전파)의 수학적 메커니즘과, 바닐라 RNN(Vanilla RNN)이 직면하는 구조적 한계점인 기울기 소실(Vanishing Gradient) 및 장기 의존성(Long-term Dependency) 문제를 심층적으로 다룹니다. 또한 이를 완화하기 위한 기법인 기울기 클리핑(Gradient Clipping)을 정리합니다.

---

#### RNN 학습 알고리즘: BPTT (Backpropagation Through Time)

##### 개념 및 작동 원리
* **BPTT 정의**: RNN은 각 시점(Time step)들이 체인(Chain) 형태로 시간 축을 따라 연결되어 있습니다. 따라서 학습 시 전체 시퀀스를 시간 축 방향으로 펼친 뒤(**Unrolling**), 오차를 시간의 역방향으로 거슬러 올라가며 기울기를 계산하고 가중치를 업데이트합니다.



##### 손실 함수 (Total Loss)
* 전체 시퀀스에 대한 최종 손실(Total Loss, $\mathcal{L}$)은 각 시점별 예측값($y_t$)과 실제 타겟값($\hat{y}_t$) 사이의 차이를 나타내는 시점별 손실 $\mathcal{L}_t$를 모두 합산하여 구합니다.

$$\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t$$

##### 가중치 공유와 기울기 합산 메커니즘
* 순환 가중치 행렬 $W_{hh}$는 모든 시점에서 완전히 동일하게 공유(Parameter Sharing)됩니다.
* 따라서 특정 시점의 손실이 $W_{hh}$에 미치는 전체 영향을 계산하려면, 현재 시점에서 $t=1$ 시점까지 모든 과거 경로를 거슬러 올라가며 연쇄 법칙(Chain Rule)을 적용하고 각 시점에서의 기여분을 전부 합산해야 합니다.

---

#### Vanilla RNN의 치명적 문제점

##### 기울기 소실 문제 (Vanishing Gradient Problem)
* **원리**: BPTT 과정에서 시간 축을 따라 역방향으로 체인 법칙을 적용할 때, 활성화 함수 $\tanh$의 미분값(최대 1 이하)과 가중치 행렬 $W_{hh}$가 반복적으로 곱해지게 됩니다.
* **결과**: 시퀀스 길이가 길어질수록 연쇄적인 곱셈에 의해 기울기가 0에 수렴하게 되며, 이로 인해 초기 시점에 입력된 정보에 대한 가중치 학습이 사실상 불가능해집니다.

##### 장기 의존성 문제 (Long-term Dependency Problem)
* **결과**: 시퀀스의 길이가 길어질수록 초기 시점의 중요 정보가 뒤로 전달되는 과정에서 점차 희석되거나 손실되어, 모델이 먼 과거의 문맥이나 단어 연관성을 기억하지 못하는 한계를 보입니다.

##### 기울기 폭발 (Exploding Gradient)
* **원리**: 반대로 가중치 행렬 $W_{hh}$의 특이값(Singular value)이 1보다 커서 연속적으로 곱해질 경우, 역방향 전파 시 기울기가 기하급수적으로 커져 무한대로 치솟습니다.
* **결과**: 모델의 학습이 극도로 불안정해지며, 수치적 오버플로우로 인해 손실값이 `NaN`(Not a Number)으로 무너지는 현상이 발생합니다.

##### 해결 방안 - 기울기 클리핑 (Gradient Clipping)
* **개념**: 기울기 폭발 문제를 해결하기 위해, 역전파 과정에서 계산된 기울기의 노름(Norm)이나 임계값(Threshold)을 설정하여 일정 수준 이상으로 커지지 않도록 강제로 잘라내는(Clipping) 기법입니다.

---

#### 공식 문서 및 참고 링크
* PyTorch Official Documentation - [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
* Deep Learning Book by Ian Goodfellow - [Chapter 10: Sequence Modeling: Recurrent and Recursive Nets](https://www.deeplearningbook.org/)
* Stanford CS230: Deep Learning - [Recurrent Neural Networks Cheatsheet](https://cs230.stanford.edu/)


