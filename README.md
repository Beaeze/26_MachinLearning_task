# YOLOv8 객체 검출 및 추적 (cat_doll, figure, pen)

직접 촬영한 사진으로 데이터셋을 구축하고 YOLOv8n으로 학습한 뒤, 영상에서 객체를 검출·추적한 프로젝트입니다.

## 클래스
- `cat_doll` (고양이 인형)
- `figure` (피규어)
- `pen` (펜)

## 개발 환경
- Google Colab (Tesla T4 GPU)
- Ultralytics YOLOv8, Roboflow (라벨링 및 데이터셋 관리)

## 데이터셋
- 스마트폰으로 직접 촬영, Roboflow에서 바운딩 박스 라벨링
- 전처리: Auto-Orient, 640×640 Resize
- 증강(Train에만 적용): 좌우 반전, 회전, 밝기, 블러 등 (3배)
- Train / Valid / Test 분할, Test 36장(42 인스턴스)
- 총 이미지 [ ]장

## 학습 설정
| 항목 | 값 |
|---|---|
| 모델 | YOLOv8n (사전학습 가중치) |
| Epochs | 50 |
| 입력 크기 | 640 |

## 성능

**검증 데이터 (50 epoch 기준)**

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.961 | 0.949 | 0.964 | 0.880 |

**테스트 데이터 (36장)**

| 클래스 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.988 | 0.917 | 0.957 | 0.868 |
| cat_doll | 0.993 | 1.000 | 0.995 | 0.948 |
| figure | 0.987 | 1.000 | 0.995 | 0.900 |
| pen | 0.985 | 0.750 | 0.882 | 0.756 |

## 영상 추적 결과
- 입력 영상: 552프레임, ByteTrack 기반 추적(`model.track`)
- cat_doll은 3개가 동시에 검출되는 구간이 많았고, figure와 pen은 간헐적으로 검출되었습니다.
- 학습 사진과 영상의 환경(배경, 거리, 각도)이 달라 테스트 성능보다 영상 검출이 불안정했습니다.

## 저장소 구성
```
.
├── README.md
├── best.pt            # 학습된 모델
├── [노트북 파일명].ipynb   # 학습 및 추적 코드 (API 키 제거)
├── results/           # 학습 그래프, 혼동 행렬, results.csv
└── video/             # 입력 영상 및 추적 결과 영상
```

## 사용법
```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("best.pt")

# 이미지 추론
model.predict(source="test.jpg", save=True, conf=0.5)

# 영상 추적
model.track(
    source="video/input_video.mp4",
    persist=True,
    tracker="bytetrack.yaml",
    conf=0.25,
    save=True,
)
```

## 참고
- 데이터셋은 Roboflow에서 직접 제작했으며, 보안상 노트북의 Roboflow API 키는 제거했습니다(`YOUR_API_KEY`).
- `best.pt`로 바로 추론과 추적을 실행할 수 있습니다.
