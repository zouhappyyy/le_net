import pickle

plan_path = ("/home/fangzheng/zoule/ESO_nnUNet_dataset/"
             "nnUNet_preprocessed/Task530_EsoTJ_30pct/"
             "nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl")

with open(plan_path, "rb") as f:
    plans = pickle.load(f)

conf = plans["plans_per_stage"][0]  # 3d_fullres

print("OLD patch size:", conf["patch_size"])

# 👇 改成 RWKV 安全尺寸
conf["patch_size"] = [64, 64, 64]

# 可选：减小 batch size（更稳）
conf["batch_size"] = 4

with open(plan_path, "wb") as f:
    pickle.dump(plans, f)

print("NEW patch size:", conf["patch_size"])

