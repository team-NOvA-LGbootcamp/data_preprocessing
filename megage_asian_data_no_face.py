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

no_face_path = "C:/Users/USER/ws/dataset/megage_asian/no_face/"
os.makedirs(no_face_path, exist_ok=True)

perfect_face_path = "C:/Users/USER/ws/dataset/megage_asian/perfect_face/"
os.makedirs(perfect_face_path, exist_ok=True)

face_detection_threshold = 0.7  # 조정 가능한 임계값

start = 20000
end = 21000

counter = 0


# 얼굴 검출 및 이미지 저장
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=face_detection_threshold) as face_detection:
    for file in image_files[start:end]:
        image = cv2.imread(file)
        counter += 1
        # 얼굴 검출 수행
        results = face_detection.process(image)
        
        # Check if any faces are detected
        if not results.detections:
            # shutil.move(file, os.path.join(no_face_path, os.path.basename(file)))
            continue  # Move to the next image if no face is detected
        else:
            shutil.move(file, os.path.join(perfect_face_path, os.path.basename(file)))
        
        if counter % 500 == 0:
            print(f"{counter}개 파일 처리 완료")
print("이미지 처리 및 저장이 완료되었습니다.")