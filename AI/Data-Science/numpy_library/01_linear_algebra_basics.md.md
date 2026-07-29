---
tags:
  - Python
  - NumPy
  - Linear_Algebra
created: 2026-07-29
---

#### 개요
머신러닝, 딥러닝 및 데이터 과학의 수치 연산과 데이터 표현에 기초가 되는 **선형대수학(Linear Algebra)**의 핵심 개념(벡터, 행렬, 특수 행렬, 선형시스템)을 정리합니다.

---

#### 벡터 (Vector)

* **정의:** 여러 숫자를 순서대로 나열한 1차원 데이터 배열로, 공간에서의 위치나 방향과 크기를 가진 양을 표현합니다.
* **차원별 수식 표현:**
  * **1차원 벡터:** $\vec{v} = \begin{bmatrix} 3 \end{bmatrix}$
  * **2차원 벡터:** $\vec{v} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$
  * **3차원 벡터:** $\vec{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$

* **기본 연산 (덧셈 및 스칼라 곱):**
  * **벡터 덧셈:** 같은 위치의 요소끼리 더합니다.
    $$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 1+3 \\ 2+4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$
  * **스칼라 곱:** 벡터의 모든 원소에 단일 숫자를 곱합니다.
    $$2 \times \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 2 \times 1 \\ 2 \times 2 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$

---

#### 행렬 (Matrix)

* **정의:** 숫자를 직사각형 수열 형태로 배열한 2차원 구조입니다.
* **의미:**
  * **데이터 표:** 행(Row)은 샘플(개체), 열(Column)은 변수(속성)를 의미합니다.
  * **선형 변환:** 공간 내 벡터를 회전, 이동, 확대/축소시키는 함수 역할을 수행합니다.

* **행렬 예시 ($A$):**
  $$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \quad (\text{3행 2열 행렬, } 3 \times 2 \text{ Matrix})$$

* **행렬 연산 규칙:**
  * **덧셈/뺄셈:** 두 행렬의 크기($m \times n$)가 완전히 동일할 때만 각 대응 원소끼리 연산 가능합니다.
  * **스칼라 곱:** 행렬 내 모든 원소에 스칼라 값을 곱합니다.
  * **행렬 곱 (Matrix Multiplication):**
    * 앞 행렬의 열 개수와 뒤 행렬의 행 개수가 일치해야 연산할 수 있습니다.
    * $(m \times n) \times (n \times p) = (m \times p)$ 크기의 행렬이 생성되며, "앞 행렬의 행 $\times$ 뒤 행렬의 열"의 내적 규칙을 따릅니다.
    $$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} (1 \times 5) + (2 \times 6) \\ (3 \times 5) + (4 \times 6) \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

#### 전치 / 단위 / 역행렬 (Special Matrices)

* **전치행렬 (Transpose Matrix, $A^T$):**
  * 행렬의 행과 열을 서로 뒤바꾼 행렬입니다.
  $$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

* **단위행렬 (Identity Matrix, $I$):**
  * 주대각선의 원소가 모두 1이고 나머지는 모두 0인 정방행렬입니다. 숫자의 '1'과 같은 역할을 하여 임의의 행렬 $A$에 단위행렬 $I$를 곱하면 항상 자기 자신이 됩니다 ($A \times I = A$).
  $$I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

* **역행렬 (Inverse Matrix, $A^{-1}$):**
  * 원래 행렬 $A$와 곱했을 때 단위행렬 $I$가 되도록 만드는 행렬입니다. (단, 행렬식 $\det(A) \neq 0$ 일 때만 존재)
  $$A \times A^{-1} = I$$

---

#### 선형시스템 (Linear Systems)

* **정의:** 여러 개의 1차 연립방정식을 행렬 곱 표현식을 사용하여 $Ax = b$ 꼴로 간결하게 나타낸 시스템입니다.
* **연립방정식 예시:**
  $$\begin{cases} 2x + y = 4 \\ x - y = 1 \end{cases}$$
* **행렬 표현 ($Ax = b$):**
  $$\begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$$
* **해구하기:** 역행렬이 존재할 경우 $x = A^{-1}b$ 연산을 통해 연립방정식의 해를 대수적으로 한 번에 구할 수 있습니다.

---

#### Python (NumPy) 코드로 보는 선형대수학 연산

```python
import numpy as np

# 1. 행렬 생성 및 전치
A = np.array([[1, 2], [3, 4]])
B = np.array([[5], [6]])

print("=== 행렬 A ===")
print(A)

print("\n=== 전치행렬 A.T ===")
print(A.T)

# 2. 행렬 곱 (Dot Product)
C = np.dot(A, B)  # 또는 A @ B
print("\n=== 행렬 곱 (A @ B) ===")
print(C)

# 3. 역행렬 계산 및 선형시스템 해(Ax = b) 구하기
# 2x + y = 4, x - y = 1 의 해 구하기
A_sys = np.array([[2, 1], [1, -1]])
b_sys = np.array([4, 1])

# np.linalg.solve 함수 활용
x_sol = np.linalg.solve(A_sys, b_sys)
print("\n=== 선형시스템 해 (x, y) ===")
print(f"x = {x_sol[0]}, y = {x_sol[1]}")
```

---

#### 공식 문서 및 참고 링크

* [NumPy Linear Algebra (numpy.linalg) Official Guide](https://numpy.org/doc/stable/reference/routines.linalg.html)
* [Khan Academy - Linear Algebra Course](https://www.khanacademy.org/math/linear-algebra)
* [3Blue1Brown - Essence of Linear Algebra (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
