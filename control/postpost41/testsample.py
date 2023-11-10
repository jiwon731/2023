import numpy as np

#file_path = '/home/user/movingcam/json/test_rempe/postpost25/postpost25_contact_info_foot.npy'
file_path = '/home/user/movingcam/data/contact_estimation/postpost01_contact_info_foot.npy'
loaded_data = np.load(file_path)

# row_index_to_modify = 118  # 수정하려는 행의 인덱스
# new_data = loaded_data  # 수정하려는 행의 새로운 데이터
#
# for i in range(20, 31):
#     new_data[i] = [1, 0]
#
# # print(new_data[118:134])
#
# # 데이터 수정
# # loaded_data[row_index_to_modify] = new_row_data
#
# # 수정된 데이터를 새로운 .npy 파일에 저장
# new_file_path = 'new_file2.npy'
# np.save(new_file_path, new_data)

print(loaded_data)