---
tags:
  - deep-learning
  - pytorch
  - autoencoder
  - cnn
  - computer-vision
  - Convolutional_Autoencoder
created: 2026-08-06
---

#### 개요

 **MNIST 데이터셋**을 활용해 이미지의 공간적 특징을 압축하고 복원하는 CNN 오토인코더(Convolutional Autoencoder)의 데이터 전처리, 아키텍처 설계, 학습 최적화 루프, 그리고 평가 및 시각화 과정을 다룹니다.

  

#### 1. 데이터셋 및 전처리 파이프라인 (Data Pipeline)

##### 1) 대상 데이터셋

- **MNIST 데이터셋**: $28 \times 28$ 크기의 흑백(Grayscale) 손글씨 숫자 이미지 데이터셋
    
      
    

##### 2) 전처리 과정 (`transforms.Compose`)

- **`transforms.ToTensor()`**: 이미지를 PyTorch Tensor 형식으로 변환하고, 픽셀 값의 범위를 기존 $[0, 255]$에서 $[0.0, 1.0]$으로 스케일링합니다.
    
      
    
- **`transforms.Normalize((0.5,), (0.5,))`**: 텐서 값의 범위를 $[0, 1]$에서 $[-1, 1]$로 정규화합니다.
    
      
    
    $$\text{Normalized Value} = \frac{x - \text{mean}}{\text{std}}$$
    

##### 3) 데이터 로더 (`DataLoader`)

- **훈련용 (`trainloader`)**: `batch_size=64`, `shuffle=True`를 적용하여 에포크마다 데이터를 무작위로 섞어 과적합을 방지합니다.
    
      
    
- **테스트용 (`testloader`)**: `batch_size=128`, `shuffle=False`를 적용하여 평가의 일관성을 유지합니다.
    
      
    

#### 2. CNN 오토인코더 모델 아키텍처 (`CNNAutoencoder`)

오토인코더는 입력을 압축하는 인코더(Encoder)와 압축된 특징으로부터 원본을 복원하는 디코더(Decoder)로 구성됩니다.

  

##### ① Encoder (인코더: 특징 압축)

- **목적**: 입력 이미지($[Batch, 1, 28, 28]$)의 공간적 구조와 핵심 특징을 유지한 채 차원을 점진적으로 축소합니다.
    
      
    
- **주요 레이어 구성**:
    
      
    - `nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)`: 흑백 이미지(채널 1)를 받아 16개의 채널로 확장합니다.
        
          
        - **`Padding = 1`의 역할**: $3 \times 3$ 필터를 훑을 때 테두리가 쪼그라드는 현상을 방지하여 이미지 크기를 $28 \times 28$로 그대로 유지합니다.
            
              
            
    - `nn.ReLU(inplace=True)`: 음수를 0으로 만드는 비선형 활성화 함수입니다. `inplace=True`를 통해 새로운 메모리를 할당하지 않고 기존 메모리를 덮어써 GPU 메모리를 절약합니다.
        
          
        
    - `nn.MaxPool2d(kernel_size=2, stride=2)`: $2 \times 2$ 영역에서 최대값을 추출하여 가로세로 크기를 절반으로 줄입니다 ($28 \times 28 \rightarrow 14 \times 14$).
        
          
        
- **잠재 공간(Latent Space)**: 두 번째 블록을 거치면서 최종적으로 **$[Batch, 32, 7, 7]$** 크기의 압축된 특징 맵을 생성합니다.
    
      
    

##### ② Decoder (디코더: 이미지 복원)

- **목적**: 인코더가 압축해 둔 잠재 공간의 특징 맵을 다시 원래의 원본 크기($28 \times 28$)로 키워내어 복원합니다.
    
      
    
- **주요 레이어 구성**:
    
      
    - `nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)`: 전치 합성곱(Upconvolution)을 통해 크기를 2배로 확대합니다 ($7 \times 7 \rightarrow 14 \times 14$, 채널은 16개로 축소).
        
          
        
    - `nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)`: 크기를 한 번 더 2배로 키워 원본 크기로 복원합니다 ($14 \times 14 \rightarrow 28 \times 28$, 채널 1개).
        
          
        
    - `nn.Sigmoid()`: 출력 픽셀 값의 범위를 $[0, 1]$ 사이로 강제 정규화하여 이미지 픽셀 스케일과 일치시킵니다.
        
          
        

#### 3. 모델 학습 및 최적화 루프 (Training Loop)

- **손실 함수 (Loss Function)**: `nn.MSELoss()` (원본 이미지와 디코더가 복원한 이미지 간의 픽셀 값 차이인 평균 제곱 오차를 비교)
    
      
    
- **옵티마이저 (Optimizer)**: `torch.optim.Adam(model.parameters(), lr=0.001)`
    
      
    
- **체크포인트 저장 로직**:
    
      
    - 에포크마다 손실(Loss) 값을 비교하여 이전보다 개선되었을 때만 로컬 서버(`cnn_autoencoder_best.pt`)에 가중치를 저장합니다.
        
          
        
    - 이후 구글 드라이브 경로(`/content/drive/MyDrive/cnn_autoencoder_best.pt`)로 안전하게 복사(`shutil.copy`)합니다.
        
          
        

#### 평가 및 특징 맵 시각화 (Evaluation & Visualization)

- **모드 전환**: `model.eval()`과 `with torch.no_grad():`를 사용하여 역전파 계산을 차단하고 평가 속도를 최적화합니다.
    
      
    
- **시각화 그리드 구조 (`Matplotlib`)**:
    
      
    - **1행 (`Original`)**: 원본 테스트 이미지 $n=10$장을 흑백(`cmap='gray'`)으로 출력합니다.
        
          
        
    - **2~7행 (`Encoded Features`)**: 전체 32개의 압축 채널 중 지정한 6개 채널(`num_channels_to_show = 6`)의 특징 맵을 `viridis` 컬러맵으로 각각 시각화하여, 모델이 숫자의 어떤 특징(선, 곡선 등)을 추출했는지 관찰합니다.
        
          
        
    - **8행 (`Decoded`)**: 디코더가 최종 복원해 낸 흑백 이미지 $n=10$장을 출력하여 원본과 비교합니다.
        
          
        

#### 핵심 개념

- **오토인코더의 손실 압축 현상**:
    
      
    - 이미지를 좁은 공간($7 \times 7$, 32채널)으로 억지로 압축(**Bottleneck**)하기 때문에, 미세한 노이즈나 거친 텍스처는 소실되고 굵직한 형태(숫자 뼈대) 위주로 매끄럽게 복원됩니다.
        
          
        
- **오토인코더 vs 세그멘테이션(U-Net) 차이**:
    
      
    - **오토인코더**: 전체 이미지를 압축했다가 원본 그대로 다시 풀어내는 방식 (압축·복원 목적, 뭉개짐 현상 발생 가능)
        
          
        
    - **세그멘테이션 / 마스킹**: 픽셀 단위의 정확한 경계선(Mask)을 따내는 방식 (인코더-디코더 사이를 다이렉트로 연결하는 U-Net 구조 등을 사용하여 디테일을 살림)
        
          
        

#### 참고 및 공식 문서 링크

- [PyTorch Official - Autoencoder Tutorial](https://pytorch.org/)
    
- [PyTorch Conv2d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    
- [PyTorch ConvTranspose2d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)