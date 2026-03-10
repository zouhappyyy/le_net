# import os
# import numpy as np
# import nibabel as nib
#
# label_dir = ("/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task601_lctsc40/labelsTr/" ) # 根据你的任务路径改
# for i, fname in enumerate(sorted(os.listdir(label_dir))):
#     if not fname.endswith(".nii.gz"):
#         continue
#     fpath = os.path.join(label_dir, fname)
#     arr = nib.load(fpath).get_fdata()
#     uniq = np.unique(arr)
#     print(fname, "unique:", uniq)
#     if i >= 5:  # 看几个就行
#         break

#
# import os
# import numpy as np
# import nibabel as nib
#
# label_dir = ("/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task601_lctsc40/labelsTr/" )   # 修改成你的真实路径
#
# for fname in sorted(os.listdir(label_dir)):
#     if not fname.endswith(".nii.gz"):
#         continue
#     fpath = os.path.join(label_dir, fname)
#     img = nib.load(fpath)
#     data = img.get_fdata()
#
#     # 显式转成整型，避免小数精度问题
#     data = data.astype(np.int16)
#
#     uniques_before = np.unique(data)
#     print(f"{fname} before:", uniques_before)
#
#     # 关键一步：255 → 1
#     data[data == 255] = 1
#
#     uniques_after = np.unique(data)
#     print(f"{fname} after:", uniques_after)
#
#     new_img = nib.Nifti1Image(data.astype(np.int16), img.affine, img.header)
#     nib.save(new_img, fpath)


import pickle

plan_path = ("/home/fangzheng/zoule/ESO_nnUNet_dataset/"
             "nnUNet_preprocessed/Task601_lctsc40/"
             "nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl")

with open(plan_path, "rb") as f:
    plans = pickle.load(f)
conf = plans["plans_per_stage"][0]  # 3d_fullres

print("OLD patch size:", conf["patch_size"])

# 👇 改成 RWKV 安全尺寸
conf["patch_size"] = [64, 64, 64]
conf["batch_size"] = 2

print("OLD patch size:", conf["patch_size"])
print("batch_size:", conf["batch_size"])