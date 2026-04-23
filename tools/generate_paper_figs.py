#!/usr/bin/env python3
"""Batch generate per-case FDConv per-band visualizations for paper figures.

This imports the visualization function from `visualize_fdconv_highfreq.py` and runs it
for a list of cases (or all .npy files in a folder). It writes images to the specified
output directory using prefixes like: <out_dir>/fdconv_<case>_slice.png

Example:
python3 tools/generate_paper_figs.py \
  --data_root /path/to/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0 \
  --max_cases 3 --device cpu --learnable_bands --output_dir ./paper_figs

"""
import os, argparse, sys
from typing import List

# ensure repo root is on path so we can import visualize script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from visualize_fdconv_highfreq import visualize_fdconv_highfreq_3d


def list_case_ids(data_root: str) -> List[str]:
    files = [f for f in os.listdir(data_root) if f.endswith('.npy')]
    files.sort()
    return [os.path.splitext(f)[0] for f in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--case_ids', nargs='*', default=None, help='List of case ids to process. If omitted, all .npy in data_root are used.')
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./paper_figs')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--k_list', nargs='+', type=int, default=[2,4,8])
    parser.add_argument('--learnable_bands', action='store_true')
    parser.add_argument('--soft_band_beta', type=float, default=30.0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.case_ids and len(args.case_ids) > 0:
        cases = args.case_ids
    else:
        cases = list_case_ids(args.data_root)

    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"Will process {len(cases)} cases, save to {args.output_dir}, device={args.device}")

    for case in cases:
        try:
            prefix = os.path.join(args.output_dir, f"fdconv_{case}")
            print(f"Processing {case} -> prefix {prefix}")
            visualize_fdconv_highfreq_3d(
                in_channels=1,
                shape=(1, 1, 32, 128, 128),
                k_list=tuple(args.k_list),
                lowfreq_att=False,
                learnable_bands=args.learnable_bands,
                soft_band_beta=args.soft_band_beta,
                device=args.device,
                save_path_prefix=prefix,
                data_root=args.data_root,
                case_id=case,
                channel=args.channel,
            )
        except Exception as e:
            print(f"Failed to process {case}: {e}")

    print('Done')

if __name__ == '__main__':
    main()
