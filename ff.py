import pickle

with open('/home/user/바탕화면/보류/res.pk', 'rb') as f:
    res = pickle.load(f)
print(res['pred_xyz_24_struct'])
xyz = res['pred_xyz_24_struct']