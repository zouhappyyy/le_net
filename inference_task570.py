import os
import argparse

from nnunet_mednext.inference.predict_simple import main as predict_simple_main


def build_args_list(
    input_dir: str,
    output_dir: str,
    task_name: str,
    trainer_class_name: str,
    plans_identifier: str,
    fold: int,
    disable_tta: bool = True,
) -> list:
    """Build argument list for predict_simple without explicit checkpoint override.

    We deliberately do NOT pass any checkpoint-related flags here so that
    predict_simple.py uses its internal default (typically model_best).
    """
    args = [
        "-i",
        input_dir,
        "-o",
        output_dir,
        "-t",
        task_name,
        "-m",
        "3d_fullres",
        "-tr",
        trainer_class_name,
        "-p",
        plans_identifier,
        "-f",
        str(fold),
    ]
    if disable_tta:
        args.append("--disable_tta")
    return args


def run_inference_for_all_models(
    images_dir: str,
    preds_root: str,
):
    os.makedirs(preds_root, exist_ok=True)

    task_name = "Task530_EsoTJ_30pct"
    fold = 1

    models = [
        (
            "Double_CCA_UPSam_fd_RWKV",
            "nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt",
            "nnUNetPlansv2.1_trgSp_1x1x1_rwkv",
        ),
        (
            "MedNeXt_S_kernel3",
            "nnUNetTrainerV2_MedNeXt_S_kernel3",
            "nnUNetPlansv2.1_trgSp_1x1x1_rwkv",
        ),
        (
            "Double_CCA_UPSam_fd_loss_RWKV",
            "nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt",
            "nnUNetPlansv2.1_trgSp_1x1x1",
        ),
    ]

    for short_name, trainer, plans in models:
        out_dir = os.path.join(preds_root, short_name, "preds")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Running inference for model {short_name} -> {out_dir}")
        argv = build_args_list(
            input_dir=images_dir,
            output_dir=out_dir,
            task_name=task_name,
            trainer_class_name=trainer,
            plans_identifier=plans,
            fold=fold,
            disable_tta=True,
        )
        cmd_str = "python -m nnunet_mednext.inference.predict_simple " + " ".join(
            f'"{a}"' if " " in a else a for a in argv
        )
        print(f"[CMD] {cmd_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Helper to print inference commands for Task570_EsoTJ83 with three trained models.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task570_EsoTJ83/imagesTr",
        help="Folder with Task570_EsoTJ83 imagesTr (nnUNet raw images)",
    )
    parser.add_argument(
        "--preds_root",
        type=str,
        default="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83",
        help="Root folder where predictions for the three models will be stored",
    )

    args = parser.parse_args()
    run_inference_for_all_models(args.images_dir, args.preds_root)


if __name__ == "__main__":
    main()

