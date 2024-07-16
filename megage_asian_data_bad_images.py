import glob
import os
import cv2
import mediapipe as mp
import numpy as np
import shutil
from tqdm import tqdm

# 밝기 계산 함수
def calculate_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_image)

# 흐림 정도 계산 함수
def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian

# Mediapipe 솔루션 초기화
mp_face_detection = mp.solutions.face_detection

# 파일 경로 설정
file_path = "C:/Users/USER/ws/dataset/megage_asian/cropped/"

# 이미지 파일 목록 가져오기 및 정렬
image_files = sorted(glob.glob(file_path + "*"))

# 출력 폴더 설정
no_face_path = "C:/Users/USER/ws/dataset/megage_asian/no_face/"
too_dark_path = "C:/Users/USER/ws/dataset/megage_asian/too_dark/"
too_blurry_path = "C:/Users/USER/ws/dataset/megage_asian/too_blurry/"


# 폴더 생성 (없는 경우)
os.makedirs(no_face_path, exist_ok=True)
os.makedirs(too_dark_path, exist_ok=True)
os.makedirs(too_blurry_path, exist_ok=True)


# 임계값 설정
face_detection_threshold = 0.01
brightness_threshold = 80
blur_threshold = 4

# 범위
start = 0
end = -1

# Mediapipe 얼굴 검출 클래스 초기화
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=face_detection_threshold) as face_detection:
    for file in tqdm(image_files[start:end], desc='Processing images', unit='image'):
        image = cv2.imread(file)
        
        # 얼굴 검출 수행
        results = face_detection.process(image)
        
        # 얼굴이 검출되지 않은 경우
        if not results.detections:
            shutil.move(file, os.path.join(no_face_path, os.path.basename(file)))
            continue
        
        
        # 밝기 평가 및 처리
        brightness = calculate_brightness(image)
        if brightness < brightness_threshold:
            shutil.move(file, os.path.join(too_dark_path, os.path.basename(file)))
            continue
        

        # 흐림 정도 평가 및 처리
        blur = calculate_blur(image)
        if blur < blur_threshold:
            shutil.move(file, os.path.join(too_blurry_path, os.path.basename(file)))
            continue

print("이미지 처리 및 저장이 완료되었습니다.")
