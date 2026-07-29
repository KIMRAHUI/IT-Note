---
tags:
  - Data_Science
  - Analysis_Process
  - Analytics_Types
created: 2026-07-29
---

#### 개요
데이터 분석(Data Analysis)의 근본적인 목적과 가치를 파악하고, 분석의 발전 단계이자 핵심 분류인 **4가지 데이터 분석 유형(기술적, 진단적, 예측적, 처방적 분석)**을 명확한 예시와 함께 학습합니다.

---

#### 데이터 분석의 목적 (Purpose of Data Analysis)

데이터 분석은 단순한 수치 정리를 넘어 비즈니스 및 연구에서 핵심적인 가치를 창출하는 것을 목표로 합니다.

* **의사결정 지원 (Decision Making):** 경험이나 직관에 의존하지 않고, 신뢰할 수 있는 데이터 근거 기반의 의사결정을 가능하게 합니다.
* **문제 해결 (Problem Solving):** 시스템 내 병목 현상, 고객 이탈, 매출 감소 등 구체적인 문제의 원인을 진단하고 해결책을 제시합니다.
* **패턴 및 인사이트 발견 (Pattern Discovery):** 방대한 데이터 속에 숨겨진 규칙성, 트렌드, 상관관계를 포착합니다.
* **예측 및 최적화 (Prediction & Optimization):** 미래 상황을 미리 파악하여 리스크를 최소화하고 자원 배분을 최적화합니다.

---

#### 데이터 분석의 4가지 유형 (4 Types of Data Analytics)

데이터 분석은 분석의 난이도와 창출되는 가치의 깊이에 따라 4단계 단계별 프레임워크로 구분됩니다.

```text
[난이도 & 가치 상승]
처방적 분석 (Prescriptive) ── "어떤 조치를 취해야 하는가?" ── [최고 가치]
      ▲
예측적 분석 (Predictive)   ── "앞으로 무슨 일이 일어날 것인가?"
      ▲
진단적 분석 (Diagnostic)   ── "왜 이런 일이 일어났는가?"
      ▲
기술적 분석 (Descriptive)  ── "무슨 일이 일어났는가?" ── [기본 분석]
```

---

##### ① 기술적 분석 (Descriptive Analytics)
* **질문:** *"무슨 일이 일어났는가?" (What happened?)*
* **특징:** 과거 데이터를 요약 및 집계하여 현재 상태와 지나온 경향을 한눈에 파악하는 가장 기본적인 분석 단계입니다. 주로 기초 통계량(평균, 표준편차), 대시보드, 보고서 형태 형태로 활용됩니다.
* **의료 예시:** *"환자의 현재 체온이 39도입니다."*
* **비즈니스 예시:** *"지난달 매출이 전월 대비 15% 감소했습니다."*

---

##### ② 진단적 분석 (Diagnostic Analytics)
* **질문:** *"왜 이런 일이 일어났는가?" (Why did it happen?)*
* **특징:** 기술적 분석 결과에서 나타난 특정 현상의 원인을 파악하기 위해 데이터를 깊이 있게 드릴다운(Drill-down)하고, 변수 간의 상관관계 및 인과관계를 추적합니다.
* **의료 예시:** *"검사 결과, 바이러스 감염 때문에 열이 났습니다."*
* **비즈니스 예시:** *"서버 장애 및 결제 페이지 오류로 인해 지난달 매출이 감소했습니다."*

---

##### ③ 예측적 분석 (Predictive Analytics)
* **질문:** *"앞으로 무슨 일이 일어날 것인가?" (What will happen?)*
* **특징:** 과거 패턴과 머신러닝, 통계적 회귀 모델을 활용하여 미래의 사건 발생 가능성이나 트렌드를 확률적으로 예측합니다.
* **의료 예시:** *"현재 추세라면 내일 열이 40도까지 오를 확률이 80%입니다."*
* **비즈니스 예시:** *"다음 달 이탈 위험이 80% 이상인 고객군은 A그룹입니다."*

---

##### ④ 처방적 분석 (Prescriptive Analytics)
* **질문:** *"최적의 결과를 위해 어떤 조치를 취해야 하는가?" (What should we do?)*
* **특징:** 예측 결과를 기반으로 목표 달성을 위한 최선의 행동 방안, 시뮬레이션, 최적화 알고리즘을 제안하는 분석의 최종 단계입니다.
* **의료 예시:** *"열을 내리기 위해 해열제 A를 500mg 투여하고 항생제 B를 함께 처방하세요."*
* **비즈니스 예시:** *"이탈 위험이 높은 A그룹 고객에게 20% 할인 쿠폰을 자동 발송하세요."*

---

#### Python 코드로 보는 4단계 분석 가상 구현 예시

```python
import pandas as pd
import numpy as np

# 1. 기술적 분석: 기초 통계 요약
print("=== 1. 기술적 분석 (Descriptive) ===")
patient_data = pd.DataFrame({'Temperature': [36.5, 37.0, 38.5, 39.0, 39.2]})
print(f"평균 체온: {patient_data['Temperature'].mean():.1f}℃")

# 2. 진단적 분석: 조건 필터링을 통한 원인 구별
print("\n=== 2. 진단적 분석 (Diagnostic) ===")
fever_patients = patient_data[patient_data['Temperature'] >= 38.0]
print(f"고열 환자 비율: {len(fever_patients)/len(patient_data)*100}% -> 감염 반응 의심")

# 3. 예측적 분석: 머신러닝/통계 기반 예측 (가상)
print("\n=== 3. 예측적 분석 (Predictive) ===")
predicted_prob = 0.80  # 80% 확률
print(f"내일 체온 40도 도달할 예상 확률: {predicted_prob*100}%")

# 4. 처방적 분석: 시뮬레이션 및 처방 룰 적용
print("\n=== 4. 처방적 분석 (Prescriptive) ===")
if predicted_prob >= 0.70:
    print("처방 제안: 해열제 500mg 즉시 투여 및 항생제 추가 처방 권장")
```

---

#### 공식 문서 및 참고 링크

* [Gartner - 4 Analytics Capabilities (Descriptive, Diagnostic, Predictive, Prescriptive)](https://www.gartner.com/en/information-technology/glossary/prescriptive-analytics)
* [Scikit-Learn Official User Guide - Machine Learning for Predictive Analytics](https://scikit-learn.org/stable/user_guide.html)
* [Harvard Business Review - Analytics Hierarchy Article](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century)
