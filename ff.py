import pickle




with open('/home/user/res.pk', 'rb') as f:
    res = pickle.load(f)
print(res['pred_xyz_24_struct'])