---
tags:
  - Deep_Learning
  - rnn
  - sequential-data
  - vanilla-rnn
  - time-series
  - nlp
  - hidden-state
  - parameter-sharing
created: 2026-08-03
---

#### 개요
본 문서는 순차 데이터(Sequential Data)의 정의와 고유한 특징, 기존 다층 퍼셉트론(MLP)이 지닌 구조적 한계, 그리고 이를 극복하기 위해 제안된 바닐라 RNN(Vanilla RNN)의 핵심 원리와 상세 수학적 수식 스펙을 다룹니다. 추가로 RNN의 다양한 입출력 유형(IO Types)과 맥락을 종합적으로 정리합니다.

---

#### 순차 데이터 (Sequential Data) 개념 및 특징

### 1) 정의
* **순차 데이터 (Sequential Data)**: 데이터의 배열 순서(Order)가 정보의 고유 의미를 결정하는 데 결정적인 역할을 하는 연속적인 데이터 형태를 의미합니다.

##### 대표적 예시
* **일일 주가 데이터**: 시계열 데이터의 대표적인 형태 (예: 1월 3일의 주가는 직전 거래일인 1월 2일의 주가 변동 영향 아래 결정됨)
* **자연어 문장 (Natural Language)**: 문장 내 단어의 순서와 전후 문맥이 의미 결정에 핵심적 역할 (예: "로미오는 줄리엣을 사랑한다"와 "줄리엣은 로미오는 사랑한다"는 완전히 다른 의미를 가짐)
* **기타 도메인**: DNA 염기 서열, 시계열 기온 변화, 샘플링된 소리 신호(Audio Signal) 등

##### 핵심 특징
* **독립성 부재**: 각 시점(Time step, $t$)의 데이터($x_t$)는 이전 시점($x_{t-1}$)의 데이터와 독립적이지 않으며 강한 상관관계를 가집니다.
* **시간 종속성 (Temporal Dependence)**: 특정 시점($t$)에서의 데이터는 과거의 모든 시점들($t_0, t_1, \dots, t_{n-1}$)의 정보와 누적된 흐름에 직접적인 영향을 받습니다.
* **가변 길이 (Variable Length)**: 시퀀스의 길이가 고정되어 있지 않고 문장이나 시계열 주기에 따라 다양하게 변화합니다.

---

#### 기존 다층 퍼셉트론(MLP)의 구조적 한계
* 일반적인 다층 퍼셉트론(MLP)은 입력 데이터 간의 순서 정보를 유기적으로 반영하거나, 고정되지 않은 가변적 길이의 입력을 처리하기 어려운 구조적 한계를 지닙니다.
* 따라서 시퀀스 구조 내부의 맥락과 순서 정보를 지속적으로 보존하며 처리할 수 있는 순환 신경망(RNN) 구조가 필수적입니다.

---

#### 바닐라 RNN (Vanilla RNN)의 핵심 원리 및 구조

##### 구조적 특징
* 단어를 한 번에 통째로 입력받는 것이 아니라 하나씩 순서대로 입력받습니다.
* 이때 이전 시점의 출력인 은닉 상태($h_{t-1}$)가 현재 입력($x_t$)과 함께 다시 입력으로 들어가 정보가 누적되는 **순환(Recurrent) 구조**를 형성합니다.

##### 파라미터 공유 (Parameter Sharing)
* 모든 시점에서 가중치($W_{xh}, W_{hh}, W_{hy}$)를 완전히 동일하게 재사용합니다.
* 이로 인해 문장의 길이가 3단어이든 100단어이든 상관없이 모델의 파라미터 개수가 일정하게 유지되는 장점을 가집니다.

---

#### RNN 핵심 공식 및 수식 스펙

##### 은닉 상태 ($h_t$) 계산 수식
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

* **$x_t \in \mathbb{R}^{d \times 1}$**: 현재 시점($t$)의 입력 벡터
* **$h_{t-1} \in \mathbb{R}^{D_h \times 1}$**: 직전 시점($t-1$)의 은닉 상태 벡터 (과거 기억)
* **$W_{xh} \in \mathbb{R}^{D_h \times d}$**: 입력-은닉 변환 가중치 행렬
* **$W_{hh} \in \mathbb{R}^{D_h \times D_h}$**: 은닉-은닉 순환 가중치 행렬
* **$b_h \in \mathbb{R}^{D_h \times 1}$**: 은닉층 편향(Bias)
* **$\tanh$**: 비선형 활성화 함수 (출력 범위를 $-1 \sim 1$로 제한하여 값의 폭발 및 소실 경향을 완화)

##### 출력값 ($y_t$) 계산 수식
$$y_t = \psi(W_{hy} h_t + b_y)$$

* **$W_{hy} \in \mathbb{R}^{D_y \times D_h}$**: 은닉-출력 가중치 행렬
* **$\psi()$**: 문제 유형에 따른 활성화 함수 (이진 분류는 Sigmoid, 다중 분류는 Softmax, 회귀는 항등함수)

---

#### RNN의 다양한 입출력 유형 (IO Types)
* **One-to-One**: 일반적인 피드포워드 신경망 (예: 이미지 분류)
* **One-to-Many**: 하나의 입력으로 시퀀스 출력 (예: 이미지 캡셔닝)
* **Many-to-One**: 시퀀스 입력으로 하나의 출력 (예: 문장 감성 분석 / 긍정·부정 분류)
* **Many-to-Many (시퀀스-투-시퀀스)**:
  * 입력 종료 후 출력 시작 (예: 기계 번역)
  * 매 시점마다 출력 (예: 비디오 프레임별 태깅)

---

#### 공식 문서 및 참고 링크
* PyTorch Official Documentation - [torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
* CS231n: Convolutional Neural Networks for Visual Recognition - [Recurrent Neural Networks Lecture Notes](https://cs231n.github.io/rnn/)
* Christopher Olah's Blog - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


