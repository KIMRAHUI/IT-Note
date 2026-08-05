---
tags:
  - deep-learning
  - pytorch
  - CNN
  - Computer-Vision
  - Data-agumentation
created: 2026-08-05
---

#### 개요

본 문서는 PyTorch를 활용한 딥러닝 이미지 분류의 기초가 되는 **CNN(Convolutional Neural Network)의 아키텍처 구조, 이미지 데이터의 수치화 및 텐서 변환, 합성곱(Convolution)과 풀링(Pooling) 레이어의 작동 원리, 주요 이미지 증강(Data Augmentation) 기법별 세부 파라미터와 주의점, 커스텀 모델 클래스 설계 기법, 그리고 학습(Training) 및 평가(Evaluation) 루프의 전체 구현 프로세스**를 정리하였습니다.

각 파트는 실행 가능한 예제 코드의 뼈대가 되며, 나중에 이 노트만 봐도 전체 파이프라인의 개념부터 구현 세부 사항까지 체계적으로 다시 학습할 수 있도록 구성했습니다.

#### Part 1. 이미지 데이터와 CNN 기초

1. **이미지 데이터의 수치화(RGB 채널과 픽셀 행렬) 구조 해석**
    
    - **상세 설명:** 컴퓨터는 이미지를 단순히 시각적인 그림으로 인지하는 것이 아니라, 높이(Height), 너비(Width), 채널(Channel)의 차원을 가진 **3차원 텐서(행렬)** 구조로 처리합니다. 흑백 이미지의 경우 채널이 1개, 컬러(RGB) 이미지의 경우 채널이 3개로 구성됩니다.
        
    - **픽셀 값 범위와 정규화:** 원본 이미지의 픽셀 값은 보통 `0부터 255` 사이의 정수형(Integer)으로 표현됩니다. 그러나 신경망 학습 시 경사 하강법(Gradient Descent)의 수렴 속도를 높이고 안정성을 확보하기 위해, 값을 `0.0 ~ 1.0` 사이의 실수형으로 나누거나 평균/표준편차를 이용해 정규화(Normalization)하는 과정을 반드시 거쳐야 합니다.
        
2. **합성곱(Convolution)과 풀링(Pooling) 레이어의 역할**
    
    - **합성곱(Conv2d):** 필터(Kernel)를 이미지 픽셀 위에서 일정한 간격(Stride)으로 슬라이딩 방식으로 이동시키며, 영역별 픽셀 값과 필터 가중치를 곱하여 모두 더하는 연산을 수행합니다. 이를 통해 이미지의 선, 모서리, 질감 등 지역적 특징(Local Features)을 효과적으로 추출해 냅니다.
        
    - **풀링(Pooling - Max / Average):** 특성 맵(Feature Map)의 공간적 크기(Spatial Dimension)를 축소하여 전체적인 연산량을 크게 줄여줍니다. 또한, 이미지 내 객체의 미세한 위치 변화나 왜곡에도 모델이 흔들리지 않고 강건함(Robustness)을 유지하도록 핵심 특징만 요약합니다.
        
3. **다차원 특성 맵을 1차원으로 펼치는 Flatten과 전결합층 연결**
    
    - 합성곱과 풀링 단계를 거치며 추출된 2차원 이상의 다차원 특성 맵은 최종 분류를 위해 `nn.Flatten()` 레이어를 통과하여 1차원 벡터 형태로 평탄화됩니다.
        
    - 변환된 1차원 벡터는 일반적인 인공신경망 계층인 전결합층(Fully Connected Layer, `nn.Linear`)으로 전달되어, 최종적으로 각 클래스별 스코어(Logits) 및 확률 값으로 매핑됩니다.
        

#### Part 2. 주요 이미지 증강(Data Augmentation) 기법, 파라미터 및 주의점

4. **전처리 및 데이터 증강 주요 클래스 (`torchvision.transforms` / `v2`)**
    
    - **`v2.RandomHorizontalFlip` (좌우 반전)**
        
        - **주요 파라미터:** `p=0.5` (적용 확률)
            
        - **목적 및 주의점:** 이미지를 무작위로 좌우 반전시켜 데이터의 다양성을 확보합니다. 단, 좌우가 바뀌어도 객체의 본래 의미와 정체성이 유지되는 데이터(일반적인 동물, 사물 등)에만 제한적으로 적용해야 합니다. (예: 글자나 의도된 방향성이 중요한 데이터에는 독이 됨)
            
    - **`v2.RandomVerticalFlip` (상하 반전)**
        
        - **주요 파라미터:** `p=0.5`
            
        - **목적 및 주의점:** 상하 반전을 통해 다양한 각도의 시각적 특징을 학습시킵니다. 단, 하늘과 땅의 위치가 고정되어 있어 뒤집히면 비상식적인 데이터가 되는 경우(예: 풍경 사진, 보행자 등)에는 절대 적용하지 않도록 주의해야 합니다.
            
    - **`v2.RandomRotation` (무작위 회전)**
        
        - **주요 파라미터:** `degrees=30` 또는 `(-15, 15)`
            
        - **목적 및 주의점:** 지정한 각도 범위 내에서 이미지를 무작위로 회전시켜 방향성에 대한 모델의 강건성을 키웁니다. 회전 시 모서리 부분에 생기는 빈 공간(테두리 영역)을 어떤 방식으로 채울지(Padding 모드 설정 등) 반드시 고려해야 합니다.
            
    - **`v2.RandomCrop` (무작위 자르기)**
        
        - **주요 파라미터:** `size=(224, 224)`
            
        - **목적 및 주의점:** 이미지의 일부 영역을 무작위로 잘라내어 객체 인식 위치 범위를 다양화하고 오버피팅을 방지합니다. 너무 과도하게 크기를 줄여 자르면 핵심 객체(중요 특징)가 잘려 나가 유실될 수 있으므로 주의가 필요합니다.
            
    - **`v2.Resize` (크기 조절)**
        
        - **주요 파라미터:** `size=(224, 224)`
            
        - **목적 및 주의점:** 입력 이미지 전체의 크기를 일정한 규격으로 일치시킵니다. 가로세로 비율(Aspect Ratio)을 고려하지 않고 강제로 크기를 조절하면 객체가 찌그러지는 왜곡이 발생하므로 비율 유지 옵션을 함께 점검해야 합니다.
            
    - **`v2.ColorJitter` (색상 및 밝기 조절)**
        
        - **주요 파라미터:** `brightness=0.2`, `contrast=0.2`, `saturation=0.2`, `hue=0.1`
            
        - **목적 및 주의점:** 이미지의 밝기, 대비, 채도, 색조를 임의로 변경하여 다양한 조명 환경과 색감 변화에 대응할 수 있도록 만듭니다. 너무 극단적인 값을 부여하면 객체 고유의 색상 특징을 완전히 가려버려 학습을 방해합니다.
            
    - **`v2.RandomErasing` (무작위 지우기 / 컷아웃)**
        
        - **주요 파라미터:** `p=0.5`, `scale=(0.02, 0.33)`
            
        - **목적 및 주의점:** 이미지의 임의의 사각형 영역을 특정 값이나 노이즈로 가려서(지워서) 일부 정보가 손상되거나 가려진 객체 환경에서도 버텨내는 능력을 부여합니다. 너무 넓은 면적을 지우면 모델이 학습할 단서 자체가 사라지므로 스케일 범위를 적절히 조절해야 합니다.
            

#### Part 3. 모델 구현 및 학습 루프 (`Cnn_Image_Classification.ipynb`)

5. **PyTorch를 이용한 커스텀 CNN 모델 클래스 설계**
    
    - **상속 구조:** PyTorch의 `nn.Module`을 상속받아 커스텀 CNN 클래스를 정의합니다.
        
    - **`__init__` 메서드:** 모델 내부에서 사용할 합성곱 층(`nn.Conv2d`), 활성화 함수(`nn.ReLU`), 풀링 층(`nn.MaxPool2d`), 그리고 최종 분류를 위한 전결합 층(`nn.Linear`) 등의 레이어 객체들을 빠짐없이 선언합니다.
        
    - **`forward` 메서드:** 입력 데이터(`x`)가 `__init__`에서 정의한 레이어들을 차례대로 통과하며 연산되는 순전파의 흐름을 직관적으로 코드로 연결합니다.
        
6. **이미지 분류 모델의 학습(Training) 및 평가(Evaluation) 루프 구현**
    
    - **데이터 로더 및 증강 적용 원칙:**
        
        - **학습 데이터(Train Dataset):** 앞서 다룬 다양한 데이터 증강(Data Augmentation) 기법들을 적극적으로 적용하여 모델의 일반화 성능을 높입니다.
            
        - **검증 및 테스트 데이터(Validation / Test Dataset):** 모델의 객관적이고 정확한 성능 평가를 위해 데이터 증강은 배제하고, 모델 학습 시와 동일한 크기 조절 및 정규화(`Normalize`) 전처리만 일관되게 적용합니다.
            
    - **순전파와 손실(Loss) 계산:**
        
        - 모델에 배치 데이터를 입력하여 순전파(Forward pass)를 진행하고 예측값(Logits)을 도출합니다.
            
        - 다중 클래스 분류(Multi-class Classification) 작업에 가장 최적화된 **`CrossEntropyLoss`** 함수를 사용하여 실제 정답 레이블과의 오차(Loss)를 정밀하게 수치화합니다.
            
    - **역전파(Backward)와 가중치 업데이트:**
        
        - 계산된 손실을 바탕으로 `optimizer.zero_grad()`를 통해 기존 그래디언트를 초기화하고, `loss.backward()`를 호출하여 오차 역전파를 통한 미분값을 계산합니다.
            
        - `optimizer.step()`을 실행하여 옵티마이저가 가중치를 일괄 업데이트하도록 제어합니다.
            
        - 학습 과정 동안 Loss(오차)와 Accuracy(정확도) 지표를 매 에폭(Epoch)마다 꾸준히 추적하고 기록하여 모델의 과적합 여부와 성능 향상 추이를 지속적으로 모니터링합니다.
            

#### 공식 문서 및 참고 링크

- [PyTorch Official - Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    
- [Torchvision Datasets & Transforms Documentation](https://pytorch.org/vision/stable/index.html)
    
- [PyTorch Transforms v2 API Reference](https://pytorch.org/vision/stable/transforms.html)
    
- [Scikit-Learn Model Evaluation Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)