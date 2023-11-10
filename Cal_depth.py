import json
import math

with open("/home/user/바탕화면/보류/postpost25/postpost25_000000000055_keypoints.json", "r") as json_file:
    data = json.load(json_file)

#정리할것
keypoint_0_x = data['people'][0]['pose_keypoints_2d'][0 * 3]
keypoint_0_y = data['people'][0]['pose_keypoints_2d'][0 * 3 + 1]
keypoint_1_x = data['people'][0]['pose_keypoints_2d'][1 * 3]
keypoint_1_y = data['people'][0]['pose_keypoints_2d'][1 * 3 + 1]
keypoint_2_x = data['people'][0]['pose_keypoints_2d'][2 * 3]
keypoint_2_y = data['people'][0]['pose_keypoints_2d'][2 * 3 + 1]
keypoint_3_x = data['people'][0]['pose_keypoints_2d'][3 * 3]
keypoint_3_y = data['people'][0]['pose_keypoints_2d'][3 * 3 + 1]
keypoint_4_x = data['people'][0]['pose_keypoints_2d'][4 * 3]
keypoint_4_y = data['people'][0]['pose_keypoints_2d'][4 * 3 + 1]
keypoint_5_x = data['people'][0]['pose_keypoints_2d'][5 * 3]
keypoint_5_y = data['people'][0]['pose_keypoints_2d'][5 * 3 + 1]
keypoint_6_x = data['people'][0]['pose_keypoints_2d'][6 * 3]
keypoint_6_y = data['people'][0]['pose_keypoints_2d'][6 * 3 + 1]
keypoint_7_x = data['people'][0]['pose_keypoints_2d'][7 * 3]
keypoint_7_y = data['people'][0]['pose_keypoints_2d'][7 * 3 + 1]
keypoint_8_x = data['people'][0]['pose_keypoints_2d'][8 * 3]
keypoint_8_y = data['people'][0]['pose_keypoints_2d'][8 * 3 + 1]
keypoint_9_x = data['people'][0]['pose_keypoints_2d'][9 * 3]
keypoint_9_y = data['people'][0]['pose_keypoints_2d'][9 * 3 + 1]
keypoint_10_x = data['people'][0]['pose_keypoints_2d'][10 * 3]
keypoint_10_y = data['people'][0]['pose_keypoints_2d'][10 * 3 + 1]
keypoint_11_x = data['people'][0]['pose_keypoints_2d'][11 * 3]
keypoint_11_y = data['people'][0]['pose_keypoints_2d'][11 * 3 + 1]
keypoint_12_x = data['people'][0]['pose_keypoints_2d'][12 * 3]
keypoint_12_y = data['people'][0]['pose_keypoints_2d'][12 * 3 + 1]
keypoint_13_x = data['people'][0]['pose_keypoints_2d'][13 * 3]
keypoint_13_y = data['people'][0]['pose_keypoints_2d'][13 * 3 + 1]
keypoint_14_x = data['people'][0]['pose_keypoints_2d'][14 * 3]
keypoint_14_y = data['people'][0]['pose_keypoints_2d'][14 * 3 + 1]


len1_01 = math.sqrt((keypoint_0_x - keypoint_1_x)**2 + (keypoint_0_y - keypoint_1_y)**2)
len1_12 = math.sqrt((keypoint_1_x - keypoint_2_x)**2 + (keypoint_1_y - keypoint_2_y)**2)
len1_23 = math.sqrt((keypoint_2_x - keypoint_3_x)**2 + (keypoint_2_y - keypoint_3_y)**2)
len1_34 = math.sqrt((keypoint_3_x - keypoint_4_x)**2 + (keypoint_3_y - keypoint_4_y)**2)
len1_15 = math.sqrt((keypoint_1_x - keypoint_5_x)**2 + (keypoint_1_y - keypoint_5_y)**2)
len1_56 = math.sqrt((keypoint_5_x - keypoint_6_x)**2 + (keypoint_5_y - keypoint_6_y)**2)
len1_67 = math.sqrt((keypoint_6_x - keypoint_7_x)**2 + (keypoint_6_y - keypoint_7_y)**2)
len1_18 = math.sqrt((keypoint_1_x - keypoint_8_x)**2 + (keypoint_1_y - keypoint_8_y)**2)
len1_89 = math.sqrt((keypoint_8_x - keypoint_9_x)**2 + (keypoint_8_y - keypoint_9_y)**2)
len1_910 = math.sqrt((keypoint_9_x - keypoint_10_x)**2 + (keypoint_9_y - keypoint_10_y)**2)
len1_1011 = math.sqrt((keypoint_10_x - keypoint_11_x)**2 + (keypoint_10_y - keypoint_11_y)**2)
len1_812 = math.sqrt((keypoint_8_x - keypoint_12_x)**2 + (keypoint_8_y - keypoint_12_y)**2)
len1_1213 = math.sqrt((keypoint_12_x - keypoint_13_x)**2 + (keypoint_12_y - keypoint_13_y)**2)
len1_1314 = math.sqrt((keypoint_13_x - keypoint_14_x)**2 + (keypoint_13_y - keypoint_14_y)**2)


with open("/home/user/바탕화면/보류/postpost25/postpost25_000000000056_keypoints.json", "r") as json_file:
    data = json.load(json_file)

keypoint_0_x = data['people'][0]['pose_keypoints_2d'][0 * 3]
keypoint_0_y = data['people'][0]['pose_keypoints_2d'][0 * 3 + 1]
keypoint_1_x = data['people'][0]['pose_keypoints_2d'][1 * 3]
keypoint_1_y = data['people'][0]['pose_keypoints_2d'][1 * 3 + 1]
keypoint_2_x = data['people'][0]['pose_keypoints_2d'][2 * 3]
keypoint_2_y = data['people'][0]['pose_keypoints_2d'][2 * 3 + 1]
keypoint_3_x = data['people'][0]['pose_keypoints_2d'][3 * 3]
keypoint_3_y = data['people'][0]['pose_keypoints_2d'][3 * 3 + 1]
keypoint_4_x = data['people'][0]['pose_keypoints_2d'][4 * 3]
keypoint_4_y = data['people'][0]['pose_keypoints_2d'][4 * 3 + 1]
keypoint_5_x = data['people'][0]['pose_keypoints_2d'][5 * 3]
keypoint_5_y = data['people'][0]['pose_keypoints_2d'][5 * 3 + 1]
keypoint_6_x = data['people'][0]['pose_keypoints_2d'][6 * 3]
keypoint_6_y = data['people'][0]['pose_keypoints_2d'][6 * 3 + 1]
keypoint_7_x = data['people'][0]['pose_keypoints_2d'][7 * 3]
keypoint_7_y = data['people'][0]['pose_keypoints_2d'][7 * 3 + 1]
keypoint_8_x = data['people'][0]['pose_keypoints_2d'][8 * 3]
keypoint_8_y = data['people'][0]['pose_keypoints_2d'][8 * 3 + 1]
keypoint_9_x = data['people'][0]['pose_keypoints_2d'][9 * 3]
keypoint_9_y = data['people'][0]['pose_keypoints_2d'][9 * 3 + 1]
keypoint_10_x = data['people'][0]['pose_keypoints_2d'][10 * 3]
keypoint_10_y = data['people'][0]['pose_keypoints_2d'][10 * 3 + 1]
keypoint_11_x = data['people'][0]['pose_keypoints_2d'][11 * 3]
keypoint_11_y = data['people'][0]['pose_keypoints_2d'][11 * 3 + 1]
keypoint_12_x = data['people'][0]['pose_keypoints_2d'][12 * 3]
keypoint_12_y = data['people'][0]['pose_keypoints_2d'][12 * 3 + 1]
keypoint_13_x = data['people'][0]['pose_keypoints_2d'][13 * 3]
keypoint_13_y = data['people'][0]['pose_keypoints_2d'][13 * 3 + 1]
keypoint_14_x = data['people'][0]['pose_keypoints_2d'][14 * 3]
keypoint_14_y = data['people'][0]['pose_keypoints_2d'][14 * 3 + 1]


len2_01 = math.sqrt((keypoint_0_x - keypoint_1_x)**2 + (keypoint_0_y - keypoint_1_y)**2)
len2_12 = math.sqrt((keypoint_1_x - keypoint_2_x)**2 + (keypoint_1_y - keypoint_2_y)**2)
len2_23 = math.sqrt((keypoint_2_x - keypoint_3_x)**2 + (keypoint_2_y - keypoint_3_y)**2)
len2_34 = math.sqrt((keypoint_3_x - keypoint_4_x)**2 + (keypoint_3_y - keypoint_4_y)**2)
len2_15 = math.sqrt((keypoint_1_x - keypoint_5_x)**2 + (keypoint_1_y - keypoint_5_y)**2)
len2_56 = math.sqrt((keypoint_5_x - keypoint_6_x)**2 + (keypoint_5_y - keypoint_6_y)**2)
len2_67 = math.sqrt((keypoint_6_x - keypoint_7_x)**2 + (keypoint_6_y - keypoint_7_y)**2)
len2_18 = math.sqrt((keypoint_1_x - keypoint_8_x)**2 + (keypoint_1_y - keypoint_8_y)**2)
len2_89 = math.sqrt((keypoint_8_x - keypoint_9_x)**2 + (keypoint_8_y - keypoint_9_y)**2)
len2_910 = math.sqrt((keypoint_9_x - keypoint_10_x)**2 + (keypoint_9_y - keypoint_10_y)**2)
len2_1011 = math.sqrt((keypoint_10_x - keypoint_11_x)**2 + (keypoint_10_y - keypoint_11_y)**2)
len2_812 = math.sqrt((keypoint_8_x - keypoint_12_x)**2 + (keypoint_8_y - keypoint_12_y)**2)
len2_1213 = math.sqrt((keypoint_12_x - keypoint_13_x)**2 + (keypoint_12_y - keypoint_13_y)**2)
len2_1314 = math.sqrt((keypoint_13_x - keypoint_14_x)**2 + (keypoint_13_y - keypoint_14_y)**2)


dis01 = len1_01**2 - len2_01**2
dis12 = len1_12**2 - len2_12**2
dis23 = len1_23**2 - len2_23**2
dis34 = len1_34**2 - len2_34**2
dis15 = len1_15**2 - len2_15**2
dis56 = len1_56**2 - len2_56**2
dis67 = len1_67**2 - len2_67**2
dis18 = len1_18**2 - len2_18**2
dis89 = len1_89**2 - len2_89**2
dis910 = len1_910**2 - len2_910**2
dis1011 = len1_1011**2 - len2_1011**2
dis812 = len1_812**2 - len2_812**2
dis1213 = len1_1213**2 - len2_1213**2
dis1314 = len1_1314**2 - len2_1314**2

dd = [dis01, dis12, dis23, dis34, dis15, dis56, dis67, dis18, dis89, dis910, dis1011, dis812, dis1213, dis1314]

result = []

for value in dd:
    if value >= 0:
        result.append(math.sqrt(value))
    else:
        result.append(-math.sqrt(abs(value)))

print(result)

# print("z 좌표 :", dis01, dis12, dis23, dis34, dis15, dis56, dis67, dis18, dis89, dis910, dis1011, dis812, dis1213, dis1314)