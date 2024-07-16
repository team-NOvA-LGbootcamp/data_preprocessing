import glob
import re
import os
import cv2
import time
from datetime import datetime
import mediapipe as mp
import numpy as np
import shutil

# Mediapipe 솔루션 초기화
mp_face_detection = mp.solutions.face_detection

# 파일 경로 설정
file_path = "C:/Users/USER/ws/dataset/megage_asian/cropped/"

# 이미지 파일 목록 가져오기 및 정렬
image_files = sorted(glob.glob(file_path + "*"))

too_dark_path = "C:/Users/USER/ws/dataset/megage_asian/too_dark/"
os.makedirs(too_dark_path, exist_ok=True)

brightness_threshold = 50  # 조정 가능한 임계값

start = 0
end = -1

counter = 0

# 밝기 계산 함수
def calculate_brightness(image):
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 밝기 계산 (평균 픽셀 값)
    return np.mean(gray_image)


# 얼굴 검출 및 이미지 저장
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    for file in image_files[start:end]:
        image = cv2.imread(file)
        counter += 1
        # 얼굴 검출 수행
        results = face_detection.process(image)
        
        # 밝기 계산
        brightness = calculate_brightness(image)
        
        # 밝기가 임계값 이하인 경우 파일 이동
        if brightness < brightness_threshold:
            shutil.move(file, os.path.join(too_dark_path, os.path.basename(file)))
        
        if counter % 500 == 0:
            print(f"{counter}개 파일 처리 완료")
        
print("이미지 처리 및 저장이 완료되었습니다.")