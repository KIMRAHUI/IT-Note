---
tags:
  - Data_Science
  - Theory
  - Data_Types
created: 2026-07-29
---

#### 개요
데이터 분석의 출발점이 되는 **데이터의 구성 요소(개체와 속성)**, **구조적 형태(정형/반정형/비정형)**, 그리고 통계 분석의 기초가 되는 **속성별 유형(데이터 척도)**을 명확하게 이해합니다.

---

#### 데이터의 구성 요소 (Data Elements)

데이터는 크게 관찰 대상이 되는 **개체**와, 그 대상이 가지고 있는 특성인 **속성**으로 구성됩니다.

* **개체 (Entity):** 분석하려는 관찰 대상 하나하나를 의미합니다.
  * *동의어:* 관찰치(Observation), 레코드(Record), 사례(Case), 샘플(Sample), 인스턴스(Instance), 행(Row)
  * *예시:* 고객 A, 특정 거래 내역 1건, 자동차 1대

* **속성 (Attribute):** 개체가 가지고 있는 측정 가능한 특성이나 변수입니다.
  * *동의어:* 변수(Variable), 특징(Feature), 차원(Dimension), 파라미터(Parameter), 열(Column)
  * *예시:* 나이, 성별, 구매 금액, 거주 지역

```python
# Pandas DataFrame 표현 예시
import pandas as pd

# 행(Index) = 개체(Entity) / 열(Columns) = 속성(Attribute)
data = {
    'Customer_ID': ['C001', 'C002', 'C003'],  # 개체 식별자
    'Age': [25, 34, 41],                    # 속성 1 (수치형)
    'Gender': ['F', 'M', 'F'],              # 속성 2 (범주형)
    'Purchase_Amount': [15000, 32000, 8900] # 속성 3 (수치형)
}
df = pd.DataFrame(data)
```

---

#### 데이터의 구조적 유형 (Structural Data Types)

데이터가 저장되고 표현되는 방식과 스키마(Schema)의 유무에 따른 분류입니다.

##### ① 정형 데이터 (Structured Data)
* **특징:** 고정된 행과 열을 가진 테이블 구조로, 엄격한 데이터 타입과 스키마가 정의되어 있어 RDBMS(SQL) 및 데이터프레임 형태로 바로 활용이 가능합니다.
* **예시:** Relational Database(MySQL, MariaDB), Excel, CSV 파일

##### ② 반정형 데이터 (Semi-structured Data)
* **특징:** 데이터 내부에 구조(메타데이터, 태그, 키-값)가 포함되어 있으나, 정형 데이터처럼 고정된 테이블 형태나 스키마를 강제하지 않는 데이터입니다.
* **예시:** JSON, XML, HTML, Log 파일

##### ③ 비정형 데이터 (Unstructured Data)
* **특징:** 정해진 구조나 규칙이 없는 데이터로, 전체 데이터의 80% 이상을 차지합니다. 전통적인 SQL로는 조회하기 어려우며 AI/머신러닝 전처리가 필수적입니다.
* **예시:** 이미지, 동영상, 음성 파일, 자유 형식 텍스트(이메일, SNS 게시글)

---

#### 데이터의 속성별 유형 (Data Measurement Scales)

데이터 분석 및 통계 모델링 시 어떤 연산(사칙연산, 평균 산출 등)을 적용할 수 있는지 결정짓는 4단계 척도 분류법입니다.

##### 정성적 데이터 / 질적 데이터 (Qualitative / Categorical Data)
숫자로 표현되어 있더라도 크기나 양을 나타내지 않고 상태나 범주를 구분하는 데이터입니다.

1. **명목형 (Nominal Scale)**
   * **특징:** 순서나 등급이 없는 단순 범주 분류.
   * **허용 연산:** 최빈값(Mode), 같음/다름(`==`, `!=`) 비교
   * **예시:** 성별(남/여), 혈액형(A/B/O/AB), 국가명

2. **순서형 (Ordinal Scale)**
   * **특징:** 범주 간에 명확한 **순서(서열)**가 존재하지만, 간격이 일정하지 않거나 수치적 차이를 계산할 수 없음.
   * **허용 연산:** 대소 비교(`>`, `<`), 중앙값(Median), 사분위수
   * **예시:** 학점(A/B/C/D), 만족도 조사(5점 척도: 매우불만~매우만족), 직급(사원-대리-과장)

#### 정량적 데이터 / 양적 데이터 (Quantitative / Numerical Data)
수치로 측정 및 계산이 가능한 데이터입니다.

1. **구간형 / 등간형 (Interval Scale)**
   * **특징:** 값들 사이의 **간격이 일정**하지만, **절대적인 0(Absolute Zero)이 존재하지 않는** 척도. (0이 '없음'을 의미하지 않음)
   * **허용 연산:** 덧셈/뺄셈(`+`, `-`), 평균(Mean), 표준편차 (곱셈/나눗셈 비율 계산 불가)
   * **예시:** 섭씨 온도(0℃는 온도가 '없는' 것이 아님), IQ 지수, 서기 연도/날짜

2. **비율형 (Ratio Scale)**
   * **특징:** **절대 영점(0 = 아무것도 없음)이 존재**하며, 간격과 비율 계산이 모두 가능한 완벽한 수치 척도.
   * **허용 연산:** 사칙연산 전체(`+`, `-`, `*`, `/`), 모든 통계량 산출 가능
   * **예시:** 키, 무게, 가격, 나이, 매출액

---

#### 공식 문서 및 참고 링크

* [Pandas Data Structures (Series & DataFrame Official Guide)](https://pandas.pydata.org/docs/user_guide/dsintro.html)
* [Scikit-Learn Preprocessing & Categorical Features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
* [Khan Academy - Quantitative & Categorical Data Types](https://www.khanacademy.org/math/ap-statistics/analyzing-categorical-ap/stats-var-types/v/identifying-individuals-variables-categorical-quantitative)
