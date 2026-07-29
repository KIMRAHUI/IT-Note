---
date: 2026-07-29
tag: [MachineLearning, Statistics, BiasVariance, Skewness, Preprocessing, EvaluationMetrics, Insight]
status: complete
---

#### 데이터 분포와 왜도 (Skewness)
* **왜도 (Skewness)의 개념**: 데이터 분포의 비대칭 정도를 나타내는 통계량으로, 데이터가 한쪽으로 치우쳐진 정도를 수치화합니다.
* **왜도의 종류**:
  * **양의 왜도 (Right-Skewed / Right-Tailed)**: 긴 꼬리가 오른쪽으로 뻗어 있는 분포입니다. ($\text{Mean} > \text{Median} > \text{Mode}$)
  * **음의 왜도 (Left-Skewed / Left-Tailed)**: 긴 꼬리가 왼쪽으로 뻗어 있는 분포입니다. ($\text{Mean} < \text{Median} < \text{Mode}$)
* **머신러닝 전처리의 필요성**: 선형 회귀 등 많은 머신러닝 알고리즘들은 데이터가 정규 분포(Normal Distribution)를 따를 때 최적의 성능을 냅니다. 따라서 왜도가 심한 데이터는 로그 변환(Log Transformation)이나 스케일링을 거쳐 대칭적인 형태로 변환해야 합니다.

---

#### 편향 (Bias)과 분산 (Variance)
* **편향 (Bias)**: 모델의 예측값이 실제 정답(True value)으로부터 얼마나 멀리 떨어져 있는가를 나타냅니다.
  * **고편향 (High Bias)**: 모델이 너무 단순하여 데이터에 내재된 패턴을 제대로 포착하지 못한 상태입니다. 이는 **과소적합(Underfit)**을 유발합니다.
* **분산 (Variance)**: 훈련 데이터의 작은 변동이나 노이즈에 모델이 얼마나 민감하게 반응하는가를 나타냅니다.
  * **고분산 (High Variance)**: 모델이 너무 복잡하여 훈련 데이터의 사소한 특징까지 과도하게 학습한 상태입니다. 이는 새로운 테스트 데이터에서 예측 실패로 이어지는 **과적합(Overfit)**을 유발합니다.

---

#### 편향과 분산의 트레이드오프 (Bias-Variance Tradeoff)
* 편향을 낮추면(모델을 복잡하게 만들면) 분산이 높아지고, 반대로 분산을 낮추면(모델을 단순하게 만들면) 편향이 높아집니다.
* **💡 인사이트**: 머신러닝 모델을 최적화한다는 것은 이 편향과 분산 사이의 최적의 균형점(Sweet spot)을 찾아내어 전체 오차(Total Error)를 최소화하는 과정입니다.

---

#### 분류 모델 평가 지표 요약 (Confusion Matrix & Metrics)
* **오차 행렬 (Confusion Matrix)**: 모델이 예측한 결과와 실제 정답을 비교하여 2차원 표로 나타낸 지표입니다 (`TP, TN, FP, FN`).
* **핵심 지표 수식**:
  * **정확도 (Accuracy)**: 
    $$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$
  * **정밀도 (Precision)**: 
    $$\text{Precision} = \frac{TP}{TP + FP}$$
  * **재현율 (Recall)**: 
    $$\text{Recall} = \frac{TP}{TP + FN}$$
  * **F1 Score (조화 평균)**: 
    $$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

#### 참고 링크 및 공식 문서
* [scikit-learn 공식 문서 - Underfitting vs. Overfitting (편향과 분산 시각화 가이드)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
* [scikit-learn 공식 문서 - Metrics and scoring (오차 행렬 및 분류 평가 지표 가이드)](https://scikit-learn.org/stable/modules/model_evaluation.html)
* [Pandas 공식 가이드 - 데이터 왜도 및 통계량 확인 (Skew)](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.skew.html)