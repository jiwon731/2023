import json
import math

with open("/home/user/바탕화면/보류/postpost25/postpost25_000000000000_keypoints.json", "r") as json_file:
    data = json.load(json_file)

keypoint_10_x = data['people'][0]['pose_keypoints_2d'][13 * 3]
keypoint_10_y = data['people'][0]['pose_keypoints_2d'][13 * 3 + 1]
keypoint_11_x = data['people'][0]['pose_keypoints_2d'][14 * 3]
keypoint_11_y = data['people'][0]['pose_keypoints_2d'][14 * 3 + 1]

# print("10번 joint 키포인트 좌표: ({}, {})".format(keypoint_10_x, keypoint_10_y))
# print("11번 joint 키포인트 좌표: ({}, {})".format(keypoint_11_x, keypoint_11_y))

len1 = math.sqrt((keypoint_10_x - keypoint_11_x)**2 + (keypoint_10_y - keypoint_11_y)**2)

# print("10~11 joint 사이 거리 :", len1)

with open("/home/user/바탕화면/보류/postpost25/postpost25_000000000001_keypoints.json", "r") as json_file:
    data = json.load(json_file)

keypoint_10_x = data['people'][0]['pose_keypoints_2d'][13 * 3]
keypoint_10_y = data['people'][0]['pose_keypoints_2d'][13 * 3 + 1]
keypoint_11_x = data['people'][0]['pose_keypoints_2d'][14 * 3]
keypoint_11_y = data['people'][0]['pose_keypoints_2d'][14 * 3 + 1]

# print("10번 joint 키포인트 좌표: ({}, {})".format(keypoint_10_x, keypoint_10_y))
# print("11번 joint 키포인트 좌표: ({}, {})".format(keypoint_11_x, keypoint_11_y))

len2 = math.sqrt((keypoint_10_x - keypoint_11_x)**2 + (keypoint_10_y - keypoint_11_y)**2)

# print(len1, len2)

dis1 = math.sqrt(len1**2 - len2**2)

print("z 좌표 :", dis1)