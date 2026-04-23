#!/usr/bin/env python3
"""Fine-tune FBM modules on center-cropped 64^3 patches to encourage high-frequency enhancement.

This script now optimizes a whole-image high-frequency share objective instead of
just maximizing raw weighted energy. The goal is to make the output spectrum
allocate more relative energy to the high-frequency region while keeping the
overall magnitude stable.
"""
import os, sys, argparse, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# avoid JIT compile during imports
_original_cpp_load = None
try:
    import torch.utils.cpp_extension as _cpp_ext
    _original_cpp_load = getattr(_cpp_ext, 'load', None)
    def _fake_load(*args, **kwargs):
        class _Dummy: pass
        return _Dummy()
    _cpp_ext.load = _fake_load
except Exception:
    _cpp_ext = None

from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import Double_RWKV_MedNeXt

# restore
try:
    if _cpp_ext is not None and _original_cpp_load is not None:
        _cpp_ext.load = _original_cpp_load
except Exception:
    pass


class NpyPatchDataset(Dataset):
    def __init__(self, data_root, patch_size=64, mode='center'):
        self.data_root = data_root
        self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.npy')]
        self.patch = patch_size
        self.mode = mode
        if len(self.files) == 0:
            raise RuntimeError('No .npy files found in '+data_root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        arr = np.load(p)
        if arr.ndim == 3:
            arr = arr[None]
        C,D,H,W = arr.shape
        if self.mode == 'center':
            sd = (D - self.patch)//2; sh = (H - self.patch)//2; sw = (W - self.patch)//2
            crop = arr[:, sd:sd+self.patch, sh:sh+self.patch, sw:sw+self.patch]
        else:
            # random crop
            sd = np.random.randint(0, D - self.patch + 1)
            sh = np.random.randint(0, H - self.patch + 1)
            sw = np.random.randint(0, W - self.patch + 1)
            crop = arr[:, sd:sd+self.patch, sh:sh+self.patch, sw:sw+self.patch]
        return torch.from_numpy(crop).float()


def _spectrum_high_ratio(vol3d, high_cut=0.30):
    """Return whole-image high-frequency energy ratio for a 3D volume."""
    fft = np.fft.rfftn(vol3d.astype(np.float32), norm='ortho')
    mag2 = np.abs(fft) ** 2
    d, h, w = vol3d.shape
    fd, fh, fw = np.meshgrid(np.fft.fftfreq(d), np.fft.fftfreq(h), np.fft.rfftfreq(w), indexing='ij')
    freq_radius = np.sqrt(fd ** 2 + fh ** 2 + fw ** 2)
    high_mask = freq_radius >= float(high_cut)
    high = float(np.sum(mag2[high_mask]))
    total = float(np.sum(mag2) + 1e-12)
    return high / total


def _torch_high_ratio(x, high_cut=0.30):
    """Differentiable-ish helper using FFT energy of the current tensor crop.

    x: [B,C,D,H,W]
    returns: scalar ratio averaged over batch/channels.
    """
    if x.dim() != 5:
        raise ValueError(f'Expected [B,C,D,H,W], got {tuple(x.shape)}')
    x_mean = x.mean(dim=1)  # [B,D,H,W]
    fft = torch.fft.rfftn(x_mean, dim=(-3, -2, -1), norm='ortho')
    mag2 = fft.real ** 2 + fft.imag ** 2
    b, d, h, w = x_mean.shape
    fd = torch.fft.fftfreq(d, device=x.device)
    fh = torch.fft.fftfreq(h, device=x.device)
    fw = torch.fft.rfftfreq(w, device=x.device)
    gfd, gfh, gfw = torch.meshgrid(fd, fh, fw, indexing='ij')
    freq_radius = torch.sqrt(gfd ** 2 + gfh ** 2 + gfw ** 2)
    high_mask = (freq_radius >= float(high_cut)).to(mag2.dtype)
    high = (mag2 * high_mask).sum(dim=(-3, -2, -1))
    total = mag2.sum(dim=(-3, -2, -1)) + 1e-12
    return (high / total).mean()


def set_fbm_requires_grad(model, requires=True):
    enc = model.encoder
    for name, param in model.named_parameters():
        param.requires_grad = False
    # enable fbm params
    for attr in ('fbm0','fbm1'):
        if hasattr(enc, attr):
            fbm = getattr(enc, attr)
            for p in fbm.parameters():
                p.requires_grad = requires


def train(args):
    device = args.device
    dataset = NpyPatchDataset(args.data_root, patch_size=64, mode='center')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # build model
    sample = np.load(dataset.files[0])
    if sample.ndim==3:
        sample = sample[None]
    in_ch = sample.shape[0]
    model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=[2,4,8], fbm_learnable=True)
    if args.ckpt and os.path.isfile(args.ckpt):
        st = torch.load(args.ckpt, map_location='cpu')
        sd = st.get('model_state', st)
        model.load_state_dict(sd, strict=False)
        print('Loaded init checkpoint', args.ckpt)
    model.to(device)
    model.train()

    # freeze everything except fbm
    set_fbm_requires_grad(model, True)

    # collect fbm params
    fbm_params = [p for n,p in model.named_parameters() if 'fbm' in n and p.requires_grad]
    if len(fbm_params) == 0:
        print('No FBM parameters found to train; abort')
        return
    optimizer = torch.optim.Adam(fbm_params, lr=args.lr, weight_decay=1e-6)

    out_dir = os.path.join('ckpt','center64')
    os.makedirs(out_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        for step, batch in enumerate(loader):
            if step >= args.steps_per_epoch:
                break
            x = batch.to(device)  # [B,C,D,H,W]
            # get encoder features without invoking RWKV heavy path if possible
            with torch.no_grad():
                try:
                    feats = model.encoder.mednext_enc.forward_encoder(x)
                except Exception:
                    feats = model.encoder(x)

            enc = model.encoder
            loss = 0.0
            hf_ratios = []
            total_energy_pen = 0.0
            for li, layer in enumerate(['fbm0','fbm1']):
                fbm = getattr(enc, layer, None)
                if fbm is None:
                    continue
                feat = feats[li]
                out, high = fbm(feat, att_feat=None, return_high=True)

                # Main objective: maximize whole-image high-frequency share of the FBM output.
                # We compute it on the crop mean across channels.
                hf_ratio = _torch_high_ratio(out)
                hf_ratios.append(hf_ratio.detach().item())

                # Stabilizer: keep overall energy close to the input feature energy.
                feat_energy = feat.pow(2).mean()
                out_energy = out.pow(2).mean()
                energy_pen = (out_energy - feat_energy).pow(2)
                total_energy_pen = total_energy_pen + energy_pen

                # Higher high-frequency ratio is better.
                loss = loss - args.hf_weight * hf_ratio

            # Regularize to avoid blow up.
            l2 = 0.0
            for p in fbm_params:
                l2 = l2 + (p**2).sum()
            loss = loss + args.energy_weight * total_energy_pen + args.l2_reg * l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % args.log_interval == 0:
                print(f'Epoch {epoch} Step {step} Loss {loss.item():.6f} hf_ratio={np.mean(hf_ratios) if hf_ratios else 0:.6f} global_step {global_step}')
        epoch_time = time.time() - epoch_start
        ckpt_path = os.path.join(out_dir, f'fbm_finetuned_epoch{epoch}.pth')
        torch.save({'model_state': model.state_dict(), 'args': vars(args)}, ckpt_path)
        print(f'Epoch {epoch} done ({epoch_time:.1f}s). Saved {ckpt_path}')
    print('Training finished. Final checkpoint in', out_dir)
    return out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0')
    parser.add_argument('--ckpt', default='tools/fd_model_ckpt_learnable.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2_reg', type=float, default=1e-6)
    parser.add_argument('--hf_weight', type=float, default=1.0, help='Weight for the high-frequency ratio maximization term')
    parser.add_argument('--energy_weight', type=float, default=10.0, help='Weight for the total-energy stabilizer term')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--log_interval', type=int, default=5)
    args = parser.parse_args()
    train(args)
