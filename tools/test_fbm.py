import torch
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FrequencyBandModulation3D

def main():
    device = 'cpu'
    x = torch.randn(1, 1, 16, 32, 32, device=device)
    fbm = FrequencyBandModulation3D(in_channels=1, k_list=[2,4,8], learnable_bands=False).to(device)
    out, high = fbm(x, return_high=True)
    print('x', x.shape)
    print('out', out.shape if out is not None else None)
    print('high', high.shape)
    print('high mean', float(high.mean()))

if __name__ == '__main__':
    main()
