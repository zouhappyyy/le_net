#!/usr/bin/env python3
"""Monitor training log for completion and generate final figures automatically.

Usage: python3 tools/monitor_and_generate.py --log tools/train_center64_100epochs.out --ckpt_dir ckpt/center64 --final_ckpt fbm_finetuned_epoch99.pth --out_dir paper_figs_trained_center64_gpu_final --poll_interval 60

The script polls for the final checkpoint file; when found, it runs the
make_annotated_figs_center64.py script to generate reviewer-facing figures that
compare the original input against the FBM-enhanced high-frequency response.
"""
import os, time, argparse, subprocess, sys

parser = argparse.ArgumentParser()
parser.add_argument('--log', default='tools/train_center64_100epochs.out')
parser.add_argument('--ckpt_dir', default='ckpt/center64')
parser.add_argument('--final_ckpt', default='fbm_finetuned_epoch99.pth')
parser.add_argument('--out_dir', default='paper_figs_trained_center64_gpu_final')
parser.add_argument('--data_root', default='/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0')
parser.add_argument('--poll_interval', type=int, default=60)
parser.add_argument('--device', default='cuda')
parser.add_argument('--gain', type=float, default=1.0, help='Scale factor passed to the figure exporter for enhanced high-frequency response')
args = parser.parse_args()

final_path = os.path.join(args.ckpt_dir, args.final_ckpt)
print(f'Monitor started: log={args.log}, final_ckpt_expected={final_path}')
# loop until final ckpt exists
while True:
    if os.path.isfile(final_path):
        print('Final checkpoint found:', final_path)
        break
    # also check for "Training finished" in log
    try:
        with open(args.log, 'r') as f:
            txt = f.read()
            if 'Training finished.' in txt or 'Training finished' in txt:
                print('Training finished message found in log')
                # try to find latest ckpt
                cands = [p for p in os.listdir(args.ckpt_dir) if p.startswith('fbm_finetuned_epoch') and p.endswith('.pth')]
                if cands:
                    # pick max epoch
                    latest = sorted(cands, key=lambda x: int(x.replace('fbm_finetuned_epoch','').replace('.pth','')))[-1]
                    final_path = os.path.join(args.ckpt_dir, latest)
                    print('Using latest checkpoint', final_path)
                    break
    except FileNotFoundError:
        pass
    time.sleep(args.poll_interval)

# generate final figures
cmd = [
    sys.executable, 'tools/make_annotated_figs_center64.py',
    '--ckpt', final_path,
    '--data_root', args.data_root,
    '--out_dir', args.out_dir,
    '--max_cases', '5',
    '--device', args.device,
    '--gain', str(args.gain),
]
print('Running make script:', ' '.join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
# stream output
for line in proc.stdout:
    print(line, end='')
proc.wait()
print('make_annotated_figs completed with exit code', proc.returncode)
print('Monitor exiting')
