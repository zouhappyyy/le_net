#!/usr/bin/env python3
"""Short training harness with FBM learnable bands enabled.
Saves checkpoint to tools/fd_model_ckpt_learnable.pth and runs analysis.
"""
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import Double_RWKV_MedNeXt
from tools.run_fd_energy_quick import load_case

OUT_CKPT = 'tools/fd_model_ckpt_learnable.pth'

def random_crop(vol, out_shape=(1,32,64,64)):
    C,D,H,W = vol.shape
    _, d,h,w = out_shape
    sd = random.randint(0, max(0, D-d))
    sh = random.randint(0, max(0, H-h))
    sw = random.randint(0, max(0, W-w))
    return vol[:, sd:sd+d, sh:sh+h, sw:sw+w]


def main():
    data_root = '/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0'
    case_id = 'ESO_TJ_60011222468'
    vol = load_case(data_root, case_id)
    print('Loaded vol', vol.shape)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_ch = vol.shape[0]
    model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=[2,4,8], fbm_learnable=True, fbm_soft_beta=30.0)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for it in range(5):
        crop = random_crop(vol, out_shape=(in_ch,32,64,64))
        x = torch.from_numpy(crop[None]).float().to(device)
        target = torch.randint(0,2,(1,32,64,64), dtype=torch.long).to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f'Iter {it} loss {loss.item()}')
    torch.save({'model_state': model.state_dict()}, OUT_CKPT)
    print('Saved checkpoint to', OUT_CKPT)
    # analyze
    os.system(f'python3 tools/analyze_trained_model.py --ckpt {OUT_CKPT}')

if __name__ == '__main__':
    main()
