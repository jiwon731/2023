import json
import cv2
import numpy as np

# JSON 파일로부터 포즈 키포인트 데이터 읽기
with open("/home/user/바탕화면/보류/postpost25/postpost25_000000000001_keypoints.json", "r") as json_file:
    data = json.load(json_file)

# OpenCV를 사용하여 이미지 생성 (임의의 크기로 설정)
image = np.zeros((720, 1280, 3), dtype=np.uint8)

# 포즈 키포인트 연결 정보 정의
connections = [
     (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13), (0, 14), (14, 16), (0, 15), (15, 17)
]

# 포즈 키포인트 선으로 연결
for connection in connections:
    start_idx, end_idx = connection
    start_point = (int(data['people'][0]['pose_keypoints_2d'][start_idx * 3]),
                  int(data['people'][0]['pose_keypoints_2d'][start_idx * 3 + 1]))
    end_point = (int(data['people'][0]['pose_keypoints_2d'][end_idx * 3]),
                int(data['people'][0]['pose_keypoints_2d'][end_idx * 3 + 1]))
    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

# 결과 이미지를 화면에 표시
cv2.imshow("Pose Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()