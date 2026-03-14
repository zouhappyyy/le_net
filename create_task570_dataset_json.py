import os
import json
import argparse
from typing import Dict, List


def discover_cases(imagesTr: str, labelsTr: str) -> List[Dict[str, str]]:
    images = []
    for fn in os.listdir(imagesTr):
        if not (fn.endswith('.nii') or fn.endswith('.nii.gz')):
            continue
        original_fn = fn
        case_id = fn
        if fn.endswith('_0000.nii.gz'):
            case_id = fn.replace('_0000.nii.gz', '')
            image_fn = f"{case_id}.nii.gz"
        elif fn.endswith('.nii.gz'):
            case_id = fn.replace('.nii.gz', '')
            image_fn = original_fn
        elif fn.endswith('.nii'):
            case_id = fn.replace('.nii', '')
            image_fn = original_fn
        else:
            image_fn = original_fn
        img_rel = os.path.join('imagesTr', image_fn).replace('\\', '/')
        # label must be <case>.nii.gz
        lbl_name = f"{case_id}.nii.gz"
        lbl_path = os.path.join(labelsTr, lbl_name)
        if not os.path.isfile(lbl_path):
            # skip images without label
            continue
        lbl_rel = os.path.join('labelsTr', lbl_name).replace('\\', '/')
        images.append({'image': img_rel, 'label': lbl_rel})
    return images


def build_dataset_json(task_root: str,
                        task_name: str = 'Task570_EsoTJ_83',
                        description: str = 'Esophagus segmentation',
                        modalities: Dict[int, str] = None,
                        labels: Dict[int, str] = None) -> Dict:
    if modalities is None:
        # single‑channel CT
        modalities = {0: 'CT'}
    if labels is None:
        # 0: background, 1: esophagus/lesion
        labels = {0: 'background', 1: 'esophagus'}

    imagesTr = os.path.join(task_root, 'imagesTr')
    labelsTr = os.path.join(task_root, 'labelsTr')
    if not os.path.isdir(imagesTr):
        raise FileNotFoundError(f'imagesTr not found: {imagesTr}')
    if not os.path.isdir(labelsTr):
        raise FileNotFoundError(f'labelsTr not found: {labelsTr}')

    training_entries = discover_cases(imagesTr, labelsTr)

    dataset = {
        'name': task_name,
        'description': description,
        'reference': 'unspecified',
        'licence': 'unspecified',
        'release': '1.0',
        'tensorImageSize': '4D',
        'modality': {str(k): v for k, v in modalities.items()},
        'labels': {str(k): v for k, v in labels.items()},
        'numTraining': len(training_entries),
        'numTest': 0,
        'training': [
            {
                'image': os.path.join('./', e['image']).replace('\\', '/'),
                'label': os.path.join('./', e['label']).replace('\\', '/')
            }
            for e in training_entries
        ],
        'test': []
    }
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate nnU-Net dataset.json for Task570_EsoTJ_83')
    parser.add_argument('--task_root', type=str,
                        default=r'E:\\ESO_nnUNet_dataset\\nnUNet_raw_data\\Task570_EsoTJ_83',
                        help='Path to Task folder containing imagesTr/labelsTr')
    parser.add_argument('--task_name', type=str,
                        default='Task570_EsoTJ_83',
                        help='Task ID / folder name as used in nnU-Net (e.g. Task570_EsoTJ_83)')

    args = parser.parse_args()

    dataset = build_dataset_json(task_root=args.task_root,
                                 task_name=args.task_name)

    out_path = os.path.join(args.task_root, 'dataset.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
    print(f'Wrote dataset.json with {dataset["numTraining"]} training cases to: {out_path}')


if __name__ == '__main__':
    main()

