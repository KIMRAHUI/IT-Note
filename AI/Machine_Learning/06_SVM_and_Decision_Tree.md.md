---
tags:
  - Machine_Learning
  - SVM
  - DecisionTree
created: 2026-07-29
---

#### 개요
판별적(Non-probabilistic) 선형 분류 알고리즘으로 마진(Margin)을 최대화하는 **Linear Support Vector Machine (Linear SVM)**과, 데이터를 계층적으로 분할하여 규칙을 생성하는 **결정 트리(Decision Tree)**의 수학적 불순도 지표(Gini, Information Gain) 및 학습 원리를 정리합니다.

---

#### Linear Support Vector Machine (Linear SVM)

* **정의:** 확률값이 아닌 카테고리를 직관적으로 구분짓는 **선형 결정 경계(Decision Boundary)**를 찾는 판별(Non-probabilistic) 모델입니다.
* **핵심 개념:**
  * **마진 (Margin):** 결정 경계선과 각 클래스별 가장 가까운 데이터 샘플 사이의 폭(거리)을 의미합니다.
  * **서포트 벡터 (Support Vector):** 마진의 경계선(Hyperplane)에 접해 있으면서 마진의 크기를 결정하는 기준이 되는 최외곽 훈련 데이터 포인트들입니다.
* **마진 최대화 전략:**
  * 데이터 구분 정확도를 확보함과 동시에 마진을 최대한 넓혀(마진 최대화) 새로운 데이터에 대한 일반화 성능을 올립니다.
* **손실 함수 (Hinge Loss):**
  * 결정 경계 바깥쪽에 정상적으로 배치된 샘플의 오차는 $0$으로 처리하고, 마진 내부나 반대편으로 침범한 오차 샘플에 대해서만 선형적으로 패널티를 부여하는 Hinge Loss를 사용합니다.
  $$\text{Cost}(h_\theta(x), y) = \begin{cases} \max(0, 1 - \theta^T x) & \text{if } y = 1 \\ \max(0, 1 + \theta^T x) & \text{if } y = 0 \end{cases}$$


---

#### 결정 트리 (Decision Tree) 및 불순도 지표

* **정의:** 나무(Tree) 구조를 기반으로 데이터에 질문(Question)을 던져가며 계층적인 의사결정 규칙을 만들어내는 분류 및 회귀 모델입니다.
* **학습 방식 (Recursive Partitioning):**
  * 각 단계마다 불순도를 가장 크게 낮추는 하나의 특성(Feature)과 임계값을 탐색하여 수직/수평의 초평면으로 반복 분할(Recursive Partitioning)을 진행합니다.

##### ① 지니 불순도 (Gini Impurity)
데이터 집단 내에 서로 다른 클래스가 얼마나 혼합되어 있는지를 나타내는 지표로, 데이터가 한 클래스로만 구성되어 순수할수록 0에 가까워집니다.

$$G(S) = 1 - \sum_{i=1}^{c} p_i^2$$


* $p_i$: 해당 데이터 세트 내에서 특정 클래스 $i$가 차지하는 비율

##### ② 정보 이득 (Information Gain)
분할 전 부모 노드의 불순도와 분할 후 자식 노드들의 가중 평균 불순도 간의 차이(불순도 감소량)를 의미하며, 결정 트리는 이 정보 이득이 최대가 되는 질문을 선택하여 분기합니다.

$$\text{Information Gain} = \text{Gini}_{\text{parent}} - \left( \frac{N_{\text{left}}}{N} \cdot \text{Gini}_{\text{left}} + \frac{N_{\text{right}}}{N} \cdot \text{Gini}_{\text{right}} \right)$$


* $\text{Gini}_{\text{parent}}$: 분할 전 부모 노드의 지니 불순도
* $N_{\text{left}}, N_{\text{right}}$: 왼쪽, 오른쪽 자식 노드의 데이터 수
* $N$: 전체 데이터 수

##### ③ 과적합 방지: 가지치기 (Pruning)
트리의 깊이(Depth)가 깊어질수록 훈련 데이터의 아주 세밀한 노이즈까지 과도하게 학습하여 오버피팅(Overfitting)이 발생합니다. 이를 방지하기 위해 트리의 최대 깊이(`max_depth`)를 제한하거나 가지치기(Pruning)를 수행해야 합니다.

---

##### Python 코드로 보는 SVM 및 Decision Tree 연산 예시

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 1. Linear SVM 모델 학습
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)

print("=== SVM Support Vector 개수 ===")
print(svm_clf.n_support_)

# 2. Decision Tree 모델 학습 (지니 불순도 기반, max_depth로 가지치기)
tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

print("\n=== Decision Tree 속성 중요도 ===")
for name, importance in zip(iris.feature_names, tree_clf.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

---

#### 공식 문서 및 참고 링크

* [ML Wiki - Support Vector Machines 이론 및 수학적 증명](http://mlwiki.org/index.php/Support_Vector_Machines)
* [Scikit-learn Official User Guide - Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
* [Scikit-learn Official User Guide - Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)