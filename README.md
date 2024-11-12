# Stereo_Depth_and_Object_Detection
 
## 자율 주행을 위한 객체 인식 및 거리 측정 모델 개발

- 참여 인원 : 박윤수, 윤병호, 차준영, 최현우
- 프로젝트 기간 : 11/6 ~ 11/12

### 프로젝트 주제

- 자율 주행 기술에서 가장 중요한 것은 도로 주행의 안정성을 높이는 것입니다. 자율 주행 차량은 다양한 환경에서 수많은 객체들(차량, 보행자 등)을 정확히 인식하고, 이들과의 거리를 정확히 측정할 수 있어야 안전한 주행이 가능합니다. 따라서 본 프로젝트는 **자율 주행 차량의 주행 안정성 강화**를 목표로, **주행 중 객체 인식 및 거리 측정을 정확히 수행할 수 있는 모델을 개발**하고자 합니다.

---

# 1. EDA

### txt file

[Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62]

[Car  |  0.00  |  0  |  1.55  |  614.24, 181.78, 727.31, 284.77  |  1.57 1.73 4.15  |  1.00 1.75 13.22  | 1.62]

| 클래스 이름 | 트렁케이션 | 오클루전 | 관찰 각도(alpha) | 2D 상자 좌표 | 3D 크기 | 3D 위치 | 회전 각도 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Car | 0.00 | 0 | 1.55 | 614.24, 181.78, 727.31, 284.77 | 1.57, 1.73, 4.15 | 1.00, 1.75, 13.22 | 1.62 |
- **클래스 이름** (`Car`): 객체 클래스
- **트렁케이션(truncation)** (`0.00`): 객체가 이미지의 경계에 의해 잘려진 부분(%). `0.0`은 잘려 있지 않음 `1.0`은 완전히 잘려 있음
- **오클루전(occlusion)** (`0`): 객체가 얼마나 가려져 있는지 나타냄.  `0`은 가려지지 않음, `1`은 부분적으로 가려짐, `2`는 상당히 가려짐, `3`은 완전히 가려짐
- **관찰 각도(alpha)** (`1.55`): 객체가 카메라에 대해 회전된 각도를 라디안 값
- **2D 경계 상자 좌표** (`614.24 181.78 727.31 284.77`): 객체 좌표. 순서대로 `xmin`, `ymin`, `xmax`, `ymax`로, 좌측 상단과 우측 하단 모서리의 x, y 좌표
- **3D 크기** (`1.57 1.73 4.15`): 객체의 실제 크기 (단위: 미터)로, 순서대로 `높이`, `너비`, `길이`
- **3D 위치** (`1.00 1.75 13.22`): 객체가 카메라로부터 떨어진 거리. `x`, `y`, `z` 위치 좌표입니다.
- **회전 각도** (`1.62`): 객체의 회전 정도의 값

## 1.1 KITTI Dataset Class Count

### 기존 Class Count

| 클래스 이름 | 개수 | 비율 (%) | 설명 |
| --- | --- | --- | --- |
| Car | 28,742 | 55.42% | 일반 승용차 |
| Van | 2,914 | 5.62% | 화물 밴 |
| Truck | 1,094 | 2.11% | 일반 트럭 |
| Pedestrian | 4,487 | 8.65% | 보행자 |
| Cyclist | 1,627 | 3.14% | 자전거 타고 있는 사람 |
| Misc | 973 | 1.88% | 기타 클래스 (특정 객체로 분류되지 않는 객체) |
| Tram | 511 | 0.99% | 도로 위 경전철 |
| DontCare | 11,295 | 21.78% | 모델 학습 시 무시해야 할 영역 및 객체 (출력되지 않음) |
| Person_sitting | 222 | 0.43% | 앉아 있는 사람 (보행자와 다른 클래스) |
| **총합** | **53,865** | **100%** |  |

### 변경된 Class Count

- Pedestrian, Person_sitting → Person
- Tram → Misc
- DontCare 삭제

| 클래스 이름 | 개수 | 비율 (%) | 설명 |
| --- | --- | --- | --- |
| Car | 28,742 | 70.85% | 일반 승용차 |
| Van | 2,914 | 7.18% | 화물 밴 |
| Truck | 1,094 | 2.7% | 일반 트럭 |
| Person | 4,709 | 11.61% | 사람 |
| Cyclist | 1,627 | 4.01% | 자전거 타고 있는 사람 |
| Misc | 1,484 | 3.66% | 기타 클래스 (특정 객체로 분류되지 않는 객체) |
| **총합** | 40,570 | 100% |  |

### 최종 Augmentation Dataset

| Class | Objects | Percentage |
| --- | --- | --- |
| Car | 55,983 | 71.01% |
| Pedestrian | 8,465 | 10.74% |
| Van | 5,831 | 7.40% |
| Misc | 1,751 | 2.22% |
| Truck | 2,386 | 3.03% |
| Cyclist | 3,020 | 3.83% |
| Person_sitting | 314 | 0.40% |
| Tram | 1,085 | 1.38% |
| **Total** | **78,835** | **100.00%** |

---

# 2. Stereo Vision (Semi-Global Block Matching)

## 2.1 Disparity(시차)

Stereo Vision은 두 개의 카메라(왼쪽, 오른쪽)를 사용하여 깊이 정보를 추정하는 기술입니다. 왼쪽 이미지와 오른쪽 이미지를 통한 동일한 객체의 위치 차이를 → “**시차”** 라고하며, 객체의 깊이를 계산합니다.

<img src="https://github.com/user-attachments/assets/cf517516-d3fe-4869-a024-babe790bda40" width=300>


<img src="https://github.com/user-attachments/assets/a4c74a5b-2bef-416f-953e-143d007cbe95" width=550>

$Where$:

$Baseline(T)$ : 두 카메라 간의 거리

$Focal Length(f)$ : 카메라 렌즈에서 이미지 센서까지의 거리 (카메라 내부 파라미터)

$Disparity(x_l - x_r)$ : 시차 (pixels)

$Depth(Z)$ : 깊이 (meters)

> 즉, 스테레오(두 개의 카메라) 비전에서 두 삼각형이 유사하면 대응하는 변의 길이 비율이 같다 라는 특성을 이용한 방법입니다
> 

### 시차 계산 Code

```python
# num_disparities=5*16 80픽셀까지의 시차
def compute_sgbm_disparity(left_image, right_image, num_disparities=5*16,
                           block_size=11, window_size=5, display=False):
    """ SGBM 알고리즘을 사용하여 완쪽 오른쪽 이미지의 시차를 계산합니다.
    """
    # P1, P2 페널티
    P1 = 8 * 3 * window_size**2
    P2 = 32 * 3 * window_size**2
    sgbm_obj = cv2.StereoSGBM_create(0, num_disparities, block_size,
        P1, P2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    # 시차 계산
    # 16배 확대하여정수로 저장된 것을 16으로 나눠서 복원
    disparity = sgbm_obj.compute(left_image, right_image).astype(np.float32)/16.0

    # 시각화
    if display:
      plt.figure(figsize = (40,20))
      plt.imshow(disparity, cmap='cividis')
      plt.title('Disparity Map', size=25)
      plt.show();

    return disparity
```

> P1 : 시차 1픽셀 차이시 패널티
> 

> P2 : 시차가 크게날시 패널티
> 
> 
> P2는 P1 보다 큰 값으로 설정하여 시차가 크게날 상황을 억제
> 

> cv2.STEREO_SGBM_MODE_SGBM_3WAY : 왼쪽-오른쪽 이미지 매칭을 세방향(왼쪽-오른쪽, 상단-하단, 대각선)으로 수행
> 

## 2.1 Resizing

### 2.1.1 640 X 640 VS 1245 X 375(원본비율)

1. **640 X 640 Depth 결과**

<img src="https://github.com/user-attachments/assets/740b41cb-5d2f-4110-9870-82806290ab5b" width=600>

<img src="https://github.com/user-attachments/assets/8acd736a-8d67-402d-acc4-4d9a841549aa" width=600>

1. **1245 X 375 (원본비율) Depth 결과**


<img src="https://github.com/user-attachments/assets/86c6cc81-f074-4eca-b6ee-60f0934e6e54" width=1000>

![image](https://github.com/user-attachments/assets/9bc0922f-104b-4c8f-b508-72d43b9e65c4)


1.  **Resizing이 Stereo Vision Disparity 안정성에 미치는 영향**
- **640 X 640**
    
<img src="https://github.com/user-attachments/assets/40a48241-dc1f-490e-927e-8df7d8f48617" width=400>
    
- **1245 X 375**

![image](https://github.com/user-attachments/assets/0fcd640f-66d5-4c47-b7db-db8bd7a462fd)


> 원본 이미지의 Bbox 비율이 Aspect Ratio와 맞지 않을 것을 고려해, 640x640 이미지로 Resizing하여 테스트하였지만, 오히려 원본 이미지에서의 disparity가 더 안정적인 것을 확인할 수 있었다.
> 

> 원본 이미지의 Bbox도 Aspect Ratio에 맞는 비율로 확인
> 

---

# 3. Object Detection Result with YOLOv8

## 3.1 YOLOv8s 모델 성능에 대한 그래프
![image](https://github.com/user-attachments/assets/a8762147-bebb-4996-9f9d-007ac3cae4ef)

![image](https://github.com/user-attachments/assets/465abc43-0fc8-4145-9485-b9f6fadd5572)

> All classes에 대해 90%이상의 높은 성능을 확인할 수 있었습니다.
> 

## 3.2 Confusion Matrix

![image](https://github.com/user-attachments/assets/2b7a4ce3-6b18-43b1-8fed-0303c4310020)


## 3.2 Epochs에 따른 성능 변화

![image](https://github.com/user-attachments/assets/0e9cfd84-00b4-45f7-b8d2-32306bfe9839)


results(loss, mAP, precision, reacall)

![image](https://github.com/user-attachments/assets/b49050fc-e764-42d6-9672-caf40e05f0e4)


각 클래스에 대한 AP, mAP 값

> 200epochs동안 Overfitting 없이 학습 회수에 따라 성능이 개선 됨을 확인 했습니다.
> 

> TP값 기준 모든 클래스 80% 이상으로 높은 성능을 확인 했습니다.
> 

---

# 4. 시연 영상

- **1245x375, 200 epochs**

[Final_2.mp4](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/b306fcf7-5801-440d-97c2-a97ef90a9d9b/Final_2.mp4)
