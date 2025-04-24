import numpy as np
import cv2
from datetime import datetime
from machine_vision.schemas import DefectModel


def INPUT_IMG(IMG_PATH):
    raw = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    return raw

def PREPROCESS_ROI(image, camera_index):
    roi_height=5730
    if camera_index == 1:
        roi_width=5480
        # camera_index가 1인 경우, 좌측부터 roi_width_index_1까지 자릅니다.
        roi_image = image[:roi_height, :roi_width]
        return roi_image
    
    else:
        width = image.shape[1]
        roi_width_index_0 = width - 5600
        roi_image = image[:roi_height, roi_width_index_0:]
        return roi_image

def PREPROCESS_PIXEL_DIFF(image: np.ndarray, group_size: int = 2) -> np.ndarray:
    """
    주어진 이미지를 세로 방향으로 group_size 픽셀씩 묶어서 각 그룹의 차이를 계산한 새로운 difference 이미지를 생성.
    
    :param image: 입력 이미지 (NumPy 배열)
    :param group_size: 묶을 픽셀 수 (세로 방향)
    :return: 차이 이미지 (원본보다 세로 크기가 줄어든 NumPy 배열)
    """
    # 이미지 크기 가져오기
    height, width = image.shape

    # group_size에 따라 나눌 수 있는 그룹 수 계산
    group_count = height // group_size

    # 차이 계산이 가능한 그룹 수는 group_count - 1
    diff_image_height = (group_count - 1) * group_size

    # 결과 이미지 생성 (원본보다 세로 크기가 줄어든 상태)
    diff_image = np.zeros((diff_image_height, width), dtype=np.uint8)

    for group_idx in range(group_count - 1):
        # 각 그룹의 시작 인덱스
        start_idx = group_idx * group_size
        next_idx = (group_idx + 1) * group_size

        # 두 그룹 간의 차이를 계산
        group_diff = cv2.absdiff(image[start_idx:start_idx+group_size, :], image[next_idx:next_idx+group_size, :])
        # 차이값을 새로운 이미지에 저장
        diff_image[start_idx:start_idx+group_size] = group_diff

    return diff_image

def PREPROCESS_BINARY(image, thr=20):
    
    _, binary_image = cv2.threshold(image, thr, 255, cv2.THRESH_BINARY)

    if binary_image.dtype != np.uint8:
        binary_image = cv2.convertScaleAbs(binary_image)

    return binary_image

def PREPROCESS_MEDIAN(image, size=3):
    # 미디언 필터를 통한 노이즈 제거
    cleaned_image = cv2.medianBlur(image, size)
    return cleaned_image

def PREPROCESS_MORPHO_CLOSE(img, kernel_size=(3,3)):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    return closing

def PREPROCESS_MORPHO_OPEN(img, kernel_size=(3,3)):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    return opening

def PREPROCESS_SOBEL(img, kernel_size=3):
    
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=int(kernel_size))
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=int(kernel_size))
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

    return sobel_combined

def PREPROCESS_GOLDEN_DIFF(roi_img, template):
    difference_image = cv2.absdiff(roi_img, template)
    return difference_image

def Preprocessing_BIG(IMG, camera_index, golden_template, size=(1024, 1024)):

    # roi_image = PREPROCESS_ROI(IMG, camera_index)
    if camera_index == 0:
        
        resize_sample = cv2.resize(IMG, size, 1)
        difference_golden = PREPROCESS_GOLDEN_DIFF(resize_sample, golden_template) # 1024, 1024 리사이즈 된 골드이미지
        sobel_golden = PREPROCESS_SOBEL(difference_golden, kernel_size=3)
        binary_golden = PREPROCESS_BINARY(sobel_golden, thr=21)
        median_filter_image = PREPROCESS_MEDIAN(binary_golden, size=5)
        close_image = PREPROCESS_MORPHO_CLOSE(median_filter_image, kernel_size=(5,5))

    else:
        resize_sample = cv2.resize(IMG, size, 1)
        difference_golden = PREPROCESS_GOLDEN_DIFF(resize_sample, golden_template)
        sobel_golden = PREPROCESS_SOBEL(difference_golden, 3)
        binary_golden = PREPROCESS_BINARY(sobel_golden, 21)
        median_filter_image = PREPROCESS_MEDIAN(binary_golden, 3)
        close_image = PREPROCESS_MORPHO_CLOSE(median_filter_image, (5,5))
    
    return resize_sample, close_image

def Preprocessing_TINY(IMG, camera_index):

    # roi_image = PREPROCESS_ROI(IMG, camera_index)
    if camera_index == 0:
        resize_image = cv2.resize(IMG, (2048, 2048), 1)
        difference_image = PREPROCESS_PIXEL_DIFF(resize_image)
        binary_image = PREPROCESS_BINARY(difference_image, thr=19)
        close_image = PREPROCESS_MORPHO_CLOSE(binary_image, kernel_size=(3,3))
        median_filter_image = PREPROCESS_MEDIAN(close_image, size=1)
        opening_image = PREPROCESS_MORPHO_OPEN(median_filter_image, kernel_size=(1,1))
        close_image = PREPROCESS_MORPHO_CLOSE(opening_image, kernel_size=(3,3))
    else:
        difference_image = PREPROCESS_PIXEL_DIFF(IMG)
        resize_image = cv2.resize(difference_image, (2048, 2048), 1)
        binary_image = PREPROCESS_BINARY(resize_image, thr=6)
        close_image = PREPROCESS_MORPHO_CLOSE(binary_image, kernel_size=(3,3))
        median_filter_image = PREPROCESS_MEDIAN(close_image, size=1)
        opening_image = PREPROCESS_MORPHO_OPEN(median_filter_image, kernel_size=(1,1))
        close_image = PREPROCESS_MORPHO_CLOSE(opening_image, kernel_size=(3,3))
    return resize_image, close_image

def PREPROCESS_CALCULATE_IOU(box1, box2):
    # box1과 box2의 좌측 상단 좌표와 너비, 높이로부터 IoU 계산
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 좌측 상단과 우측 하단 좌표 계산
    x1_min = x1
    y1_min = y1
    x1_max = x1 + w1
    y1_max = y1 + h1

    x2_min = x2
    y2_min = y2
    x2_max = x2 + w2
    y2_max = y2 + h2

    # 교차 영역 계산
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union != 0 else 0

def PREPROCESS_MERGE_BOX(box1, box2):
    # 박스의 좌측 상단과 크기를 이용하여 병합된 박스 계산
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 두 박스를 포함하는 최소 좌측 상단 좌표와 최대 우측 하단 좌표 계산
    merged_x_min = min(x1, x2)
    merged_y_min = min(y1, y2)
    merged_x_max = max(x1 + w1, x2 + w2)
    merged_y_max = max(y1 + h1, y2 + h2)
    
    # 병합된 박스의 좌측 상단 좌표와 크기 계산
    merged_w = merged_x_max - merged_x_min
    merged_h = merged_y_max - merged_y_min
    
    return (merged_x_min, merged_y_min, merged_w, merged_h)

def PREPROCESS_OVERLAPPING_BOX(A, B):
    # A와 B 리스트를 합침
    boxes = A + B # 4 + 19 = 23

    # 병합된 박스가 더 이상 겹치는 것이 없을 때까지 반복
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(boxes) - 1:
            j = i + 1
            while j < len(boxes):
                iou = PREPROCESS_CALCULATE_IOU(boxes[i], boxes[j])
                
                if iou > 0:
                    # 겹치면 두 박스를 병합
                    merged_box = PREPROCESS_MERGE_BOX(boxes[i], boxes[j])
                    
                    # 병합된 박스를 리스트에 넣고 기존 두 박스를 제거
                    boxes[i] = merged_box
                    del boxes[j]
                    
                    # 병합이 일어났으므로 다시 시작
                    merged = True
                else:
                    j += 1
            i += 1

    # 리스트 안의 모든 원소를 튜플로 반환
    return [tuple(box) for box in boxes]

def MEASUREMENTS_ONLY_COUNT(big_target_preprocessed_img, tiny_target_preprocessed_img):
    
    count = 0
    # 큰 이미지 처리
    cnt_big, _, stats_big, _ = cv2.connectedComponentsWithStats(big_target_preprocessed_img)

    for i in range(1, cnt_big):  # 배경 제외, 객체 정보만 처리
        (_, _, _, _, area) = stats_big[i]

        # 노이즈 제거
        if area < 96:
            continue
        else:
            count += 1

    # 작은 이미지 처리
    
    cnt_tiny, _, stats_tiny, _ = cv2.connectedComponentsWithStats(tiny_target_preprocessed_img)

    for i in range(1, cnt_tiny):  # 배경 제외, 객체 정보만 처리
        (_, _, _, _, area) = stats_tiny[i]

        # 노이즈 제거
        if area < 96:
            continue
        else:
            count += 1

    return count

def MEASUREMENTS(big_target_roi_img, big_target_preprocessed_img, tiny_target_roi_resize_img, tiny_target_preprocessed_img):
    
    tiny_boxes = []
    big_boxes = []  # 큰 이미지의 박스를 저장할 리스트
    output_boxes = list[DefectModel]()
    
    cnt_big, _, stats_big, _ = cv2.connectedComponentsWithStats(big_target_preprocessed_img)
    results_img = cv2.cvtColor(big_target_roi_img, cv2.COLOR_GRAY2BGR) # 1024, 1024

    big_boxes = []  # 큰 이미지의 박스를 저장할 리스트

    for i in range(1, cnt_big):  # 배경 제외, 객체 정보만 처리
        (x, y, w, h, area) = stats_big[i]

        # 노이즈 제거
        if area < 5:
            continue

        # 큰 이미지 크기에 맞게 좌표 변환
        print(area)
        x_resized = int(x)
        y_resized = int(y)
        w_resized = int(w)
        h_resized = int(h)

        # 변환된 박스 저장
        big_boxes.append((x_resized, y_resized, w_resized, h_resized))
    print("큰 객체 탐지 개수 : ", len(big_boxes))

    # 작은 이미지 처리 (tiny_target_rawsize_img = 2048x2048, tiny_target_preprocessed_img = 가변 크기)
    tiny_height, tiny_width = tiny_target_preprocessed_img.shape[:2]  # 작은 이미지의 전처리된 크기 # 2048, 2048
    orig_height, orig_width = big_target_roi_img.shape[:2]  # 원본 작은 이미지 크기 (2048x2048) # 1024, 1024

    # 작은 이미지 스케일링 비율 계산
    scale_x_tiny = orig_width / tiny_width # 1024/2048
    scale_y_tiny = orig_height / tiny_height # 1024/2048

    cnt_tiny, _, stats_tiny, center = cv2.connectedComponentsWithStats(tiny_target_preprocessed_img)

    tiny_boxes = []  # 작은 이미지의 박스를 저장할 리스트

    for i in range(1, cnt_tiny):  # 배경 제외, 객체 정보만 처리
        (x, y, w, h, area) = stats_tiny[i]

        # 노이즈 제거
        if area < 18:
            continue

        # 원본 작은 이미지 크기에 맞게 좌표 변환
        x_resized = int(x * scale_x_tiny)
        y_resized = int(y * scale_y_tiny)
        w_resized = int(w * scale_x_tiny)
        h_resized = int(h * scale_y_tiny)

        tiny_boxes.append((x_resized, y_resized, w_resized, h_resized))
    print("작은 객체 탐지 개수 : ", len(tiny_boxes))

    # 두 박스 세트를 합치고 병합
    last_boxes = PREPROCESS_OVERLAPPING_BOX(big_boxes, tiny_boxes)
    count = len(last_boxes)
    for i, (x, y, w, h) in enumerate(last_boxes):
        cv2.rectangle(results_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        output_boxes.append(DefectModel(idx=i, x1=x, y1=y, x2=x+w, y2=y+h))
        
    return count, results_img, output_boxes

def YOLO_DETECTION_PREPROCESS(image, box, target_size=(640, 640)):
    """
    객체 크기에 따라 640x640 이미지를 생성합니다.
    
    Args:
    - image: 원본 이미지 (numpy 배열 형식)
    - box: (x, y, w, h) 형식의 박스 (중심 좌표 기준)
    - target_size: 최종 리사이즈 크기 (기본값 640x640)
    
    Returns:
    - 640x640으로 리사이즈된 3채널 이미지 (numpy 배열)
    """
    x_center, y_center, w_box, h_box = box

    # 객체 크기의 1.5배를 계산
    w_box_scaled = w_box * 1.5
    h_box_scaled = h_box * 1.5

    # Case 1: 객체의 1.5배 크기가 320보다 작을 때
    if w_box_scaled < 320 and h_box_scaled < 320:
        crop_width = max(320, w_box_scaled)
        crop_height = max(320, h_box_scaled)
        
        x_min = int(x_center - crop_width / 2)
        y_min = int(y_center - crop_height / 2)
        x_max = int(x_center + crop_width / 2)
        y_max = int(y_center + crop_height / 2)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        cropped_image = image[y_min:y_max, x_min:x_max]

        # 3채널로 변환
        if len(cropped_image.shape) == 2:
            cropped_image = np.stack((cropped_image,) * 3, axis=-1)

        # 패딩을 추가하여 640x640 만들기
        padded_image = cv2.copyMakeBorder(
            cropped_image, 
            (target_size[1] - cropped_image.shape[0]) // 2, 
            (target_size[1] - cropped_image.shape[0]) // 2, 
            (target_size[0] - cropped_image.shape[1]) // 2, 
            (target_size[0] - cropped_image.shape[1]) // 2, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )

    # Case 2: 1.5배 크기가 320보다 크고 640 이하일 때
    elif w_box_scaled <= 640 and h_box_scaled <= 640:
        crop_width = int(w_box_scaled)
        crop_height = int(h_box_scaled)
        
        x_min = int(x_center - crop_width / 2)
        y_min = int(y_center - crop_height / 2)
        x_max = int(x_center + crop_width / 2)
        y_max = int(y_center + crop_height / 2)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        cropped_image = image[y_min:y_max, x_min:x_max]

        # 3채널로 변환
        if len(cropped_image.shape) == 2:
            cropped_image = np.stack((cropped_image,) * 3, axis=-1)

        # 패딩을 추가하여 640x640 만들기
        padded_image = cv2.copyMakeBorder(
            cropped_image, 
            (target_size[1] - cropped_image.shape[0]) // 2, 
            (target_size[1] - cropped_image.shape[0]) // 2, 
            (target_size[0] - cropped_image.shape[1]) // 2, 
            (target_size[0] - cropped_image.shape[1]) // 2, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )

    # Case 3: 640보다 큰 경우
    else:
        # 3채널로 변환
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        # 640x640 크기로 리사이즈
        padded_image = cv2.resize(image, target_size)

    return padded_image

def YOLO_DETECTION_PREPROCESS_MULTI_IMGS(image, boxes, target_size=(640, 640)):
    """
    여러 박스 정보를 처리하여 각각 640x640 이미지를 생성합니다.
    
    Args:
    - image: 원본 이미지 (numpy 배열 형식)
    - boxes: 여러 개의 (x, y, w, h) 형식의 박스 (중심 좌표 기준) 리스트
    - target_size: 최종 리사이즈 크기 (기본값 640x640)

    Returns:
    - 640x640 크기의 여러 이미지들 (리스트로 반환)
    """
    result_images = []
    
    for index, box in enumerate(boxes):
        # 각 박스마다 학습용 이미지 생성
        processed_image = YOLO_DETECTION_PREPROCESS(image, box, target_size)
        result_images.append(processed_image)
    
    return result_images

# def preprocess(IMAGE, golden_template):

    # # 1. RoI 처리
    # roi_image = crop_image_RoI(IMAGE, camera_index=1) # 탑재시에는 해당 함수 비활성화
    # 2. 이미지 축소
    resize_image = cv2.resize(IMAGE, (1024, 1024), 1)
    # 3. difference IMG
    difference_image = cv2.absdiff(resize_image, golden_template)
    # 4. Sobel Filter(배경요소 제거)
    sobel_x = cv2.Sobel(difference_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(difference_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
    # 4. 이진화 처리
    binary_image = image_binary(sobel_combined, thr=21)
    # 5. median filter 
    median_filter_image = remove_noise_median(binary_image)
    # 6. 이미지 패딩
    # padding_edge_image = apply_internal_padding(median_filter_image, 20)
    # 구조화 요소 커널(갖고 있는 불량 이미지에 따라 값 조절 필요)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 닫힘 연산 적용 ---③
    closing = cv2.morphologyEx(median_filter_image, cv2.MORPH_CLOSE, k)


    return resize_image, closing

def PREPROCESS_golden(IMAGE, camera_index, size):

    # # 1. RoI 처리
    roi_image = PREPROCESS_ROI(IMAGE, camera_index)
    # 2. 이미지 축소
    resize_image = cv2.resize(roi_image, size, 1)

    return resize_image

def GET_golden_image(camera_index, size=(1024, 1024)):
    if camera_index == 1:
        golden_image_path = "machine_vision/images/golden_template_cam1(20250304)_mean_resize.png"
        resize_golden_template = cv2.imread(golden_image_path, 0)
        # resize_golden_template = PREPROCESS_golden(raw_golden_image, camera_index, size) # 1024, 1024
    else:
        golden_image_path = "machine_vision/images/golden_template_cam0(20250304)_mean_resize.png"
        resize_golden_template = cv2.imread(golden_image_path, 0) # 1024, 1024
        # resize_golden_template = PREPROCESS_golden(raw_golden_image, camera_index, size) # 27일 버전은 이미 리사이즈와 RoI가 적용된 상태라서 비활성화

    return resize_golden_template

def process_yolo_results(results, class_names):
    """
    YOLO 모델 결과를 DataFrame으로 정리합니다.

    Args:
    - results: YOLO 모델의 예측 결과 (ultralytics YOLOv8 모델 결과 객체)
    - class_names: 클래스 번호와 이름의 매핑 (예: model.names)

    Returns:
    - DataFrame: 탐지된 정보가 정리된 DataFrame
    """
    data = []

    for result in results:
        if len(result.boxes) == 0:
            # 탐지된 객체가 없을 경우
            data.append([None, -1, None, -1, None, -1, result.speed['inference']])
            continue

        # 탐지된 클래스 ID 리스트
        class_ids = result.boxes.cls.tolist()
        
        # 클래스별 탐지 개수 카운트
        class_counter = Counter(class_ids)
        
        # 가장 많이 탐지된 클래스 (가장 첫 번째 클래스)
        most_common = class_counter.most_common(3)  # 최대 3개 클래스의 (클래스 ID, 개수) 추출
        
        # 클래스 ID를 label 이름으로 변환 및 None/-1 처리
        processed_classes = []
        for i in range(3):
            if i < len(most_common):
                class_id, count = most_common[i]
                class_label = class_names[class_id]  # 클래스 이름으로 변환
                processed_classes.extend([class_label, count])
            else:
                processed_classes.extend([None, -1])  # 없는 경우 None과 -1로 채움

        # DataFrame에 추가할 데이터 정리
        row = processed_classes + [result.speed['inference']]
        data.append(row)

    # DataFrame 생성
    columns = [
        'Most Detected Class', 'Most Detected Class Count', 
        '2nd Detected Class', '2nd Detected Class Count', 
        '3rd Detected Class', '3rd Detected Class Count', 
        'Inference Time'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    return df

# # 클래스 번호와 이름 매핑 예시 (모델로부터 가져올 수 있음)
# # 예: model.names = {0: 'person', 1: 'bicycle', 2: 'car', ...}
# class_names = model.names

# # YOLO 예측 수행 (detection_result_images는 YOLO 모델에 입력된 이미지 리스트)
# YOLO_res = model(detection_result_images)

# # 결과를 DataFrame으로 변환
# df_yolo_results = process_yolo_results(YOLO_res, class_names)

# # 결과 확인
# print(df_yolo_results)

def GENERATE_filename(camera_index, is_result=False):
    # 현재 시간을 yyyyMMddHHmmssfff 형식으로 포맷팅
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    
    if is_result == True:
        filename = f"{timestamp}-{camera_index}_result.bmp"    
    
    else:
        # 파일 이름 생성
        filename = f"{timestamp}-{camera_index}.bmp"
    
    return filename