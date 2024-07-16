import glob
import re
import os
import cv2
import time
from datetime import datetime
import mediapipe as mp

# Mediapipe 솔루션 초기화
mp_face_detection = mp.solutions.face_detection

# 파일 경로 설정
test_image_dir_path = "C:/Users/USER/ws/dataset/megage_asian/megaage_asian/test/"
train_image_dir_path = "C:/Users/USER/ws/dataset/megage_asian/megaage_asian/train/"
test_label_path = "C:/Users/USER/ws/dataset/megage_asian/test_age2.txt"
train_label_path = "C:/Users/USER/ws/dataset/megage_asian/train_age2.txt"
save_path = "C:/Users/USER/ws/dataset/megage_asian/cropped/"

os.makedirs(save_path, exist_ok=True)

# 파일명에서 숫자 추출하여 정렬
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

# 이미지 파일 목록 가져오기 및 정렬
test_image_files = sorted(glob.glob(test_image_dir_path + "*"), key=extract_number)
train_image_files = sorted(glob.glob(train_image_dir_path + "*"), key=extract_number)
image_files = test_image_files + train_image_files

# 레이블 파일 읽기
with open(test_label_path, 'r', encoding='utf-8') as file:
    test_labels = file.readlines()
with open(train_label_path, 'r', encoding='utf-8') as file:
    train_labels = file.readlines()

# 레이블 리스트 생성
labels = [line.strip() for line in test_labels + train_labels]

start = 0
end = -1

counter = 0
fail = []

# 얼굴 검출 및 이미지 저장
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    for file, label in zip(image_files[start:end], labels[start:end]):
        image = cv2.imread(file)
        counter += 1
        if counter%500 == 0:
            print(counter)
        # 얼굴 검출 수행
        results = face_detection.process(image)
        
        # 검출된 얼굴이 있는 경우
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                if bboxC is None:
                    continue
                
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                if x < 0 or y < 0 or w > iw or h > ih:
                    continue
                
                detected_image = image[y:y+h, x:x+w]
                age, gender = label.split()

                # gender 재설정 (1 -> 0, 0 -> 1)
                gender = '0' if gender == '1' else '1'

                # 타임스탬프 포함 파일명 생성
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(int(time.time() * 1000000))
                filename = f"{age}_{gender}_2_{timestamp}.jpg"

                # 이미지 저장
                cv2.imwrite(os.path.join(save_path, filename), cv2.resize(detected_image, (200, 200)))
        else:
            fail +=[file]
        
print("이미지 처리 및 저장이 완료되었습니다.")

print("실패")
for file in fail:
    print(file)