---
tags:
  - Deep_Learning
  - rnn
  - nlp
  - word-embedding
  - one-hot-encoding
  - embedding-layer
created: 2026-08-03
---

#### 개요
본 문서는 자연어 처리(NLP)의 기본 개념과 컴퓨터가 언어를 이해하기 위해 필수적인 단어 표현 방식(레이블 인코딩, 원-핫 인코딩, 임베딩 벡터)의 특징 및 한계점을 비교 분석합니다. 추가로 임베딩 레이어의 동작 원리와 학습 방법에 대해 체계적으로 정리합니다.

---

#### 자연어 처리 (NLP, Natural Language Processing) 개요

##### 1) 개념
* **자연어 처리(NLP)**: 인간의 언어 의미를 분석하여 컴퓨터가 처리할 수 있도록 변환하고 응용하는 인공지능 분야입니다.

##### 2) 핵심 과제
* 컴퓨터는 문자 그 자체를 직접 이해할 수 없으므로, 모든 언어 요소를 **숫자(수치형 벡터)**로 변환해야 합니다.

##### 3) 기본 전처리 과정
1. **토큰화 (Tokenization)**: 문장을 단어, 형태소, 서브워드 등 최소 의미 단위로 분할
2. **정수 인코딩 (Integer Encoding)**: 각 토큰에 고유한 정수 ID 부여
3. **패딩 (Padding)**: 배치 연산을 위해 시퀀스 길이를 동일하게 맞춤

---

#### 단어 표현 방식의 비교 (Label vs One-Hot vs Embedding)

| 구분 | Label Encoding | One-Hot Encoding | Embedding Vector (Dense) |
| :--- | :--- | :--- | :--- |
| **표현 방식** | 단어마다 고유 정수 부여<br>(예: Red=0, Green=1) | 해당 인덱스만 1, 나머지는 0인 희소 벡터<br>(Sparse Matrix) | 의미를 학습한 고차원 실수 밀집 벡터<br>(Dense Matrix) |
| **행렬 형태** | 1차원 정수 | 희소 행렬 (Sparse Matrix) | 밀집 행렬 (Dense Matrix) |
| **차원 및 메모리** | 1차원 (효율적) | 단어 사전 크기만큼 차원 폭발<br>(메모리 비효율적) | 고정된 적절한 밀집 차원 사용<br>(메모리 효율적) |
| **단어 간 관계** | 표현 불가 | 표현 불가<br>(모든 단어 간 거리가 동일/직교) | 표현 가능<br>(의미가 유사하면 벡터 거리가 가까움) |

---

#### 원-핫 인코딩(One-Hot Encoding)의 2가지 치명적 문제점

1. **차원의 폭발 문제 (Curse of Dimensionality)**
   * 단어 사전(Vocabulary)의 크기가 10만 개이면, 단어 하나를 표현하기 위해 10만 차원의 벡터가 필요하며 대부분의 값이 0으로 채워져 메모리가 크게 낭비됩니다.
2. **의미 다루기 불가 (Lack of Semantic Relationship)**
   * 모든 원-핫 벡터 간의 내적(Dot Product) 값이 0이 되므로(직교성), 단어 간의 연관성이나 유사성(예: '사과'와 '바나나')을 전혀 표현할 수 없습니다.

---

#### 임베딩 레이어 (Embedding Layer) 동작 및 학습 방법

##### 1) 임베딩 (Embedding)의 개념
* 단어나 문장을 저차원~고차원 실수 벡터 공간 상의 점으로 매핑하여 단어 간 의미적 유사성을 표현하는 기술입니다.
* **예시**: 벡터 공간 상에서 `"apple"`과 `"banana"`는 서로 가까운 위치에, `"apple"`과 `"car"`는 멀리 떨어진 위치에 배치됩니다.

##### 2) 임베딩 벡터 학습 방법 2가지
1. **사전 훈련된 임베딩 (Pre-trained Embeddings)**
   * Word2Vec, GloVe 등 거대 데이터셋으로 미리 학습된 가중치를 가져와 활용함으로써 학습 속도를 단축하고 일반화 성능을 높입니다.
2. **훈련 중 직접 학습 (Learned During Training)**
   * 풀고자 하는 특정 Task의 손실(Loss)을 최소화하는 방향으로 신경망 내부의 임베딩 레이어 가중치를 직접 경사 하강법으로 업데이트합니다.

---

#### 공식 문서 및 참고 링크

* [PyTorch Official Documentation - torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
* [Gensim Documentation - Gensim Word2Vec Model Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
* [Mikolov et al. (2013) - Efficient Estimation of Word Representations in Vector Space (Word2Vec Paper)](https://arxiv.org/abs/1301.3781)

