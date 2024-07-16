from os import listdir
from os.path import isfile, join
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

dir_path = " "
IMAGE_FILES = [dir_path+"/"+f for f in listdir(dir_path) if isfile(join(dir_path, f))]

with mp_face_detection.FaceDetection(
  model_selection=0, min_detection_confidence=0.5) as face_detection:
  
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)

    # 작업 전에 BGR 이미지를 RGB로 변환
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    results = face_detection.process(image)

    # 검출된 얼굴이 있는 경우
    if results.detections:
      for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        
        if bboxC is None:
          break
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        if x < 0 or y < 0 or w > iw or h > ih:
          break
       
        detected_image = image[y:y+h, x:x+w]
        print(idx, file, print(len(results.detections)))
        cv2.imwrite(f"./results/{file[11:]}", cv2.resize(detected_image, (200, 200)))

    # BGR 이미지로 변환하여 화면에 출력
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('MediaPipe Face Detection', image)
    # cv2.waitKey(0)