---
tags:
  - Machine_Learning
  - Preprocessing
  - FeatureScaling
  - CrossValidation
created: 2026-07-29
---

#### 개요
머신러닝 모델의 학습 안정성과 예측 성능을 확보하기 위한 핵심 전처리 단계인 **피처 스케일링(Feature Scaling)**의 필요성과 종류, 모델의 일반화 상태를 판가름하는 **과적합/과소적합(Overfitting/Underfitting)**, 그리고 학습 데이터만을 활용해 객관적인 성능 평가를 수행하는 **K-Fold 교차 검증(Cross Validation)**을 상세히 정리합니다.

---

#### Feature Scaling (피처 스케일링)의 필요성

* **정의:** 머신러닝 모델에 입력할 각 변수(Feature)들의 수치 범위(Scale)를 일치시키거나 분포를 균일하게 정규화하는 전처리 작업입니다.
* **스케일링이 필수적인 이유:**
  * **거리 기반 모델 (KNN, SVM, K-Means 등):** 변수 간 스케일 차이가 크면, 값이 상대적으로 큰 특성이 유클리디안 거리 연산을 지배하여 스케일이 작은 특성이 무시됩니다.
  * **경사하강법 기반 모델 (Linear/Logistic Regression, Neural Networks 등):** 특성별 스케일이 다르면 손실 함수의 등고선이 길게 찌그러진 타원형 구조가 되어, 경사하강법의 수렴 속도가 매우 느려지고 불안정하게 진동합니다.

#### 주요 스케일링 기법 수식 및 특징

1. **Min-Max Scaling (정규화, Normalization)**
   * 모든 데이터를 $0 \sim 1$ 범위(또는 지정한 최솟값~최댓값)로 변환합니다.
   * 데이터의 최소/최대 위치를 고정하므로 이상치(Outlier)에 매우 민감합니다.
   $$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

2. **Standardization (표준화, Z-Score Scaling)**
   * 데이터를 평균 $\mu = 0$, 표준편차 $\sigma = 1$인 표준정규분포 형태로 변환합니다.
   * 데이터의 상한/하한이 정해지지 않으며 정규분포를 가정하는 선형 모델에 유용합니다.
   $$x' = \frac{x - \mu}{\sigma}$$

3. **Robust Scaling**
   * 평균과 표준편차 대신 **중앙값(Median)**과 **사분위수 범위(IQR, $Q_3 - Q_1$)**를 활용합니다.
   * 이상치(Outlier)의 영향을 최소화하면서 데이터를 변환하는 강건성을 가집니다.
   $$x' = \frac{x - Q_2(\text{median})}{Q_3 - Q_1}$$

---

#### 과적합(Overfitting)과 과소적합(Underfitting)

머신러닝 모델 학습 과정에서 편향(Bias)과 분산(Variance)의 트레이드오프 관계를 나타내는 핵심 개념입니다.

```text
[과소적합 (Underfitting)]        [적정 적합 (Good Fit)]        [과적합 (Overfitting)]
   모델이 너무 단순함              일반화 성능 확보             모델이 너무 복잡함
(Train 낮음, Test 낮음)         (Train 높음, Test 높음)      (Train 높음, Test 낮음)
```

* **과소적합 (Underfitting):**
  * 모델이 지나치게 단순하여 훈련 데이터에 내재된 규칙이나 패턴조차 제대로 학습하지 못한 상태입니다.
  * **증상:** 훈련 데이터(Train)와 검증 데이터(Validation) 모두 예측 성능이 낮습니다.
* **과적합 (Overfitting):**
  * 모델이 지나치게 복잡하여 훈련 데이터의 사소한 노이즈나 노이즈성 무작위 특성까지 과도하게 암기한 상태입니다.
  * **증상:** 훈련 데이터 점수는 매우 높으나, 한 번도 보지 못한 테스트 데이터에 대한 일반화 성능이 현저히 떨어집니다.
* **해결책:**
  * 모델 복잡도 조절 (트리 깊이 제한, 가중치 수 축소)
  * 데이터 추가 확보 및 노이즈 제거
  * **규제(Regularization)** 적용 (L1/L2 Penalty)

---

#### 교차 검증 (Cross Validation)과 K-Fold

* **목적:** 테스트 데이터(Test Set)는 최종 성능 평가 및 배포 단계용으로 완전히 아껴두고, 훈련 데이터(Train Set)만을 여러 방식으로 분할하여 모델의 일반화 성능을 객관적으로 검증하기 위함입니다.
* **K-Fold 교차 검증 작동 원리:**
  1. 전체 훈련 데이터를 크기가 같은 $K$개의 폴드(Fold) 세트로 분할합니다.
  2. 첫 번째 루프에서는 1번째 폴드를 검증 데이터(Validation Set)로, 나머지 $(K-1)$개 폴드를 훈련 데이터로 활용하여 학습 및 평가합니다.
  3. 검증 폴드의 위치를 바꿔가며 $K$번 반복 학습 및 평가를 진행합니다.
  4. $K$개의 평가 결과를 최종 평균 내어 대표 성능 지표로 사용합니다.

```text
Fold 1    Fold 2    Fold 3    ...    Fold K
[ Validation ] [   Train    ] [   Train    ] ... [   Train    ] -> Iteration 1
[   Train    ] [ Validation ] [   Train    ] ... [   Train    ] -> Iteration 2
...
[   Train    ] [   Train    ] [   Train    ] ... [ Validation ] -> Iteration K
```

---

##### Python 코드로 보는 스케일링 및 교차 검증 구현 예시

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 1. Feature Scaling 적용 (StandardScaler 예시)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-Fold 교차 검증 객체 생성 (5-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. 모델 정의 및 교차 검증 연산 수행
model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')

print("=== K-Fold 교차 검증 결과 ===")
print("각 Fold별 정확도:", scores)
print(f"평균 정확도: {scores.mean():.4f}")
```

---

#### 공식 문서 및 참고 링크

* [Hands-On Machine Learning (Aurélien Géron) GitHub Notebooks](https://github.com/ageron/handson-ml3)
* [Scikit-learn Official User Guide - Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
* [Scikit-learn Official User Guide - Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
