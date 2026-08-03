---
tags:
  - Deep_Learning
  - rnn
  - lstm
  - gru
  - cell-state
  - gates
  - sequence-modeling
created: 2026-08-03
---

#### 개요
본 문서는 바닐라 RNN(Vanilla RNN)의 치명적인 한계점인 **기울기 소실(Vanishing Gradient)과 장기 의존성(Long-term Dependency)** 문제를 극복하기 위해 제안된 고급 순차 모델인 **LSTM (Long Short-Term Memory)과 이를 더욱 경량화·간소화한 GRU (Gated Recurrent Unit)의 아키텍처 원리, 상세 수식, 그리고 두 모델 간의 비교를 심층적으로 다룹니다.

---

#### LSTM (Long Short-Term Memory)

##### 개념 및 구조적 특징
* **개념**: RNN의 단일 은닉 상태 통로 구조에 장기 기억을 전담하는 **Cell State ($C_t$)**를 추가하고, 3개의 게이트(Gate)를 도입하여 정보의 흐름을 선택적으로 제어하는 구조입니다.
* **핵심 통로**:
  * **Cell State ($C_t$)**: 마치 컨베이어 벨트처럼 네트워크 전체를 관통하며 장기 기억을 운반합니다 (덧셈 위주의 연산이 주로 이루어져 역전파 시 기울기 소실 문제를 효과적으로 방지).
  * **Hidden State ($h_t$)**: 단기 기억 및 최종 출력을 담당합니다.


##### LSTM 내부 아키텍처 구성 요소 (마크다운 표 정리)

| 구성 요소 | 종류 / 명칭 | 핵심 역할 및 설명 |
| :--- | :--- | :--- |
| **핵심 상태 통로** | Cell State ($C_t$) | 네트워크를 관통하며 장기 기억을 운반하는 고속도로 역할 |
| | Hidden State ($h_t$) | 단기 기억 및 이번 시점의 최종 출력을 담당 |
| **3대 게이트** | 망각 게이트 (Forget Gate, $f_t$) | 이전 장기 기억($C_{t-1}$) 중 어떤 정보를 지울지 결정 ($\sigma$, 범위 $0 \sim 1$) |
| | 입력 게이트 (Input Gate, $i_t, \tilde{C}_t$) | 현재 입력 중 무엇을 새 장기 기억에 저장할지 결정 (세기 $i_t$와 후보값 $\tilde{C}_t$ 결합) |
| | 출력 게이트 (Output Gate, $o_t$) | 갱신된 장기 기억($C_t$)을 바탕으로 단기 기억($h_t$)으로 무엇을 내보낼지 결정 |

---

#### 3대 게이트(Gate) 수식 및 상세 수식 스펙

##### ① 망각 게이트 (Forget Gate, $f_t$)
* **역할**: 이전 시점의 장기 기억($C_{t-1}$) 중 어떤 정보를 지우고 어떤 정보를 보존할지 결정합니다 ($0 \sim 1$ 사이의 값 출력).
* **수식**:
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

##### ② 입력 게이트 (Input Gate, $i_t, \tilde{C}_t$)
* **역할**: 현재 입력된 정보 중 어떤 내용을 새로운 장기 기억에 반영할지 결정합니다.
* **수식**:
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
  $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad (\text{현재 입력에 따른 후보 장기 기억})$$

##### ③ Cell State 갱신
* **역할**: 과거의 기억($C_{t-1}$) 중 일부를 잊고($f_t \odot C_{t-1}$), 새로운 기억($i_t \odot \tilde{C}_t$)을 더하여 최종 장기 기억($C_t$)을 완성합니다.
* **수식**:
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad (\odot \text{은 Hadamard Product})$$

##### ④ 출력 게이트 (Output Gate, $o_t$ 및 $h_t$)
* **역할**: 갱신된 장기 기억($C_t$)을 $\tanh$ 함수로 정규화한 뒤, 출력 게이트($o_t$)의 제어를 받아 이번 시점의 단기 기억이자 출력인 $h_t$를 생성합니다.
* **수식**:
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
  $$h_t = o_t \odot \tanh(C_t)$$

---

#### GRU (Gated Recurrent Unit)

#####  개념 및 구조적 특징
* **개념**: LSTM의 복잡한 3개 게이트와 별도의 Cell State 구조를 대폭 간소화하여, 내부 상태를 $h_t$ 하나로 통합하고 게이트를 2개로 줄인 모델입니다.
* **특징**: LSTM에 비해 파라미터 수가 적어 연산 속도가 빠르고, 상대적으로 데이터가 적은 환경에서도 과적합을 방지하는 데 유리합니다.

##### GRU 내부 아키텍처 구성 요소 (마크다운 표 정리)

| 구성 요소 | 종류 / 명칭 | 핵심 역할 및 설명 |
| :--- | :--- | :--- |
| **핵심 상태 통로** | Hidden State ($h_t$) | LSTM의 장/단기 상태를 하나로 통합한 단일 은닉 상태 통로 |
| **2대 게이트** | 리셋 게이트 (Reset Gate, $r_t$) | 새로운 후보 은닉 상태를 만들 때 과거 기억($h_{t-1}$)을 얼마나 참고할지 결정 |
| | 업데이트 게이트 (Update Gate, $z_t$) | 과거 기억 유지 비율($1-z_t$)과 새 정보 반영 비율($z_t$)을 시소처럼 동시 조절 |

---

#### 2대 게이트 수식 및 역할

##### ① 리셋 게이트 (Reset Gate, $r_t$)
* **역할**: 새로운 후보 은닉 상태를 계산할 때, 과거의 기억($h_{t-1}$)을 얼마나 무시하거나 참고할지 결정합니다.
* **수식**:
  $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

##### ② 업데이트 게이트 (Update Gate, $z_t$)
* **역할**: LSTM의 Forget Gate와 Input Gate의 역할을 동시에 수행하며, 과거 기억을 유지하는 비율($(1-z_t)$)과 새로운 정보를 반영하는 비율($z_t$)을 시소처럼 동시에 조절합니다.
* **수식**:
  $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

##### ③ 후보 은닉 상태 및 최종 은닉 상태($h_t$) 갱신
* **후보 은닉 상태 수식**:
  $$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$
* **최종 은닉 상태($h_t$) 수식**:
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

---

#### LSTM vs GRU 아키텍처 비교 요약

| 비교 항목 | LSTM (Long Short-Term Memory) | GRU (Gated Recurrent Unit) |
| :--- | :--- | :--- |
| **상태 통로 (State)** | $h_t$ (단기) + $C_t$ (장기) 총 2개 | $h_t$ 하나로 통합 (1개) |
| **게이트 종류** | Forget, Input, Output (3개) | Reset, Update (2개) |
| **파라미터 및 학습** | 상대적으로 많음 (대규모 데이터 및 복잡한 패턴 학습에 유리) | 상대적으로 적음 (학습 속도가 빠르고 과적합 방지에 유리) |

---

#### 공식 문서 및 참고 링크
* PyTorch Official Documentation - [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
* PyTorch Official Documentation - [torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
* Hochreiter & Schmidhuber (1997) - [Long Short-Term Memory Original Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)



