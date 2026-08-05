---
tags:
  - deep-learning
  - pytorch
  - tensor
  - MOC
  - DataModeling
created: 2026-08-04
---
---

#### 개요

본 문서는 PyTorch의 핵심 데이터 구조인 **텐서(Tensor)의 행렬 구조 해석, 생성 함수, CPU/GPU 장치(Device) 관리 및 디바이스 불일치 에러 해결법, 데이터 타입 변환, NumPy 배열과의 연동, 그리고 브로드캐스팅(Broadcasting) 원리를 심층적으로 다룹니다.

각 노트는 원문 내용에 더해 **실행 가능한 예제 코드, 예상 출력, 왜 그런 결과가 나오는지에 대한 원리 설명, 자주 하는 실수**까지 포함해서 나중에 이 노트만 봐도 처음부터 다시 학습할 수 있도록 구성했습니다.


#### Part 1. 텐서(Tensor) 다루기

1. [Tensor-Shape-Interpretation](./Pytorch_Tensor/Tensor-Shape-Interpretation.md) — 텐서의 행과 열(Shape) 해석 방법
2. [[Tensor-Attributes-Cheatsheet]] — 텐서 기본 속성 및 메서드 치트시트
3. [[Tensor-Creation-Functions]] — 특수한 텐서 생성 함수
4. [[Dtype-Conversion-NumPy-Interop]] — 데이터 타입 변환 및 NumPy 연동
5. [[Device-Management-CPU-GPU]] — CPU/GPU 장치 관리 및 에러 해결
6. [[Tensor-Operations-Broadcasting]] — 텐서 연산 및 브로드캐스팅

#### Part 2. 모델 만들기 (torch.nn)

7. [[Autograd-Gradient-Computation]] — Autograd와 Gradient 계산 (backward, requires_grad, no_grad)
8. [[Torch-nn-Basics]] — torch.nn 기초 (Linear, 활성화 함수, 손실 함수, 텐서 vs 레이어 차원)
9. [[nn-Module-Model-Building]] — nn.Module 상속, nn.Sequential, 복잡한 모델 설계, 모델 정보 확인
10. [[Model-Device-Management]] — 모델을 CPU/GPU로 옮기기

#### Part 3. 모델 학습시키기

11. [[Training-Loop-and-Optimizer]] — Training Loop, Loss 계산, Optimizer, 모델 저장/불러오기
12. [[In-place-Methods-and-Functional]] — in-place 메소드, torch.nn.functional

#### 참고

13. [[References]] — 공식 문서 및 참고 링크

#### 전체 개념 

|노트|핵심 질문|
|---|---|
|[[Tensor-Shape-Interpretation]]|shape `(2, 3)`을 보면 행/열을 어떻게 읽는가?|
|[[Tensor-Attributes-Cheatsheet]]|지금 이 텐서의 상태(형태/타입/장치/미분추적)를 어떻게 확인하는가?|
|[[Tensor-Creation-Functions]]|랜덤값/특정값으로 텐서를 어떻게 새로 만드는가?|
|[[Dtype-Conversion-NumPy-Interop]]|텐서의 데이터 타입을 어떻게 바꾸고, NumPy와 어떻게 오가는가?|
|[[Device-Management-CPU-GPU]]|CPU/GPU 사이에서 텐서를 어떻게 이동하고, 불일치 에러는 어떻게 해결하는가?|
|[[Tensor-Operations-Broadcasting]]|크기가 다른 텐서끼리 연산이 어떻게 자동으로 맞춰지는가?|
|[[Autograd-Gradient-Computation]]|그래디언트는 어떻게 자동 계산되고, 언제 꺼야 하는가?|
|[[Torch-nn-Basics]]|레이어/활성화 함수/손실 함수를 어떻게 만드는가?|
|[[nn-Module-Model-Building]]|나만의 모델 클래스를 어떻게 설계하는가? (단순/복잡/중첩 구조)|
|[[Model-Device-Management]]|모델 전체를 어떻게 GPU로 옮기는가?|
|[[Training-Loop-and-Optimizer]]|모델을 실제로 어떻게 학습시키고 저장/평가하는가?|
|[[In-place-Methods-and-Functional]]|`_`가 붙은 메서드와 `F.xxx()` 함수는 무엇이 다른가?|
