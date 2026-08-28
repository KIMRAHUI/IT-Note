---
tags: [MOC, object-detection, yolo, ssd]
---

# Object Detection


## 학습 순서

| 순서 | 노트 | 핵심 키워드 |
|:--:|:--|:--|
| 1 | [[03_YOLOv1]] | Grid 회귀, IoU, NMS, Loss 4요소 |
| 2 | [[02_SSD]] | Default Box, 다중 스케일, torchvision 실전 파이프라인 |
| 3 | [[04_YOLOv2]] | Anchor box, Darknet-19, Passthrough(Space-to-Depth) |
| 4 | [[05_YOLOv3]] | Darknet-53, Residual Block, FPN 스타일 3-스케일 |
| 5 | [[06_참고_ultralytics_YOLO]] | 실전 라이브러리 활용, 비디오 추론, CLI 학습 |

## 세대별 구조 비교

| 구분 | YOLOv1 | YOLOv2 | YOLOv3 | SSD |
|:--|:--|:--|:--|:--|
| Backbone | 커스텀 24-conv | Darknet-19 | Darknet-53 | ResNet(가변) |
| 예측 방식 | 그리드 직접 회귀 | Anchor + 단일 스케일(13x13) | Anchor + 3-스케일(FPN식) | Anchor + 6-스케일(독립) |
| 작은 객체 대응 | 취약 | Passthrough로 일부 개선 | 3-스케일+업샘플 concat으로 대폭 개선 | 초반 저층 feature 활용 |
| 클래스 예측 | Softmax류(MSE) | Softmax류 | **Sigmoid(독립 로지스틱, multi-label)** | Softmax |
| 실전 구현 난이도 | 직접 구현 쉬움 | 중간 | 중간~높음 | torchvision 활용 시 쉬움 |

## 공통 핵심 개념 링크

- **IoU / NMS**: [[03_YOLOv1#3-iou-intersection-over-union]], [[03_YOLOv1#4-nms-non-maximum-suppression]]
- **Anchor Box**: [[04_YOLOv2#3-yolov2-전체-모델-백본--passthrough--detection-head]], [[02_SSD#2-모델--앵커-생성자anchor-generator-초기화]]
- **Multi-Scale 검출**: [[05_YOLOv3#4-yolov3-전체-모델-backbone--fpn-스타일-neck--3-head]], [[02_SSD#1-ssd-백본backbone-정의]]
- **Loss 설계 패턴**(좌표 sqrt 보정 + obj/noobj 가중치 분리): 03/04/05 노트 공통


