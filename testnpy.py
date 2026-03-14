import os
import argparse
import numpy as np


def convert_npz_dir_to_npy(
    input_dir: str,
    output_dir: str,
    key: str,
    overwrite: bool = False,
) -> None:
    """
    将一个目录下所有 .npz 文件转换为 .npy 文件.

    参数:
        input_dir: 包含 .npz 文件的目录
        output_dir: 输出 .npy 文件的目录（可以和 input_dir 相同）
        key: 从 .npz 中取数组时使用的 key
             - 如果为 None，则尝试:
               1) 如果只有一个数组，直接取那个唯一的 key
               2) 否则优先 "data", "arr_0" 这样的常见 key
        overwrite: 如果目标 .npy 已存在，是否覆盖
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
    files.sort()
    if not files:
        print(f"No .npz files found in {input_dir}")
        return

    print(f"Found {len(files)} .npz files in {input_dir}")
    for fn in files:
        in_path = os.path.join(input_dir, fn)
        base = os.path.splitext(fn)[0]  # 去掉 .npz
        out_path = os.path.join(output_dir, base + ".npy")

        if os.path.exists(out_path) and not overwrite:
            print(f"[SKIP] {out_path} already exists (use --overwrite to force)")
            continue

        try:
            data = np.load(in_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                # 多数组容器
                keys = list(data.keys())
                if key is not None:
                    if key not in keys:
                        raise KeyError(
                            f"Key '{key}' not found in {in_path}, available keys: {keys}"
                        )
                    arr = data[key]
                else:
                    # 未指定 key 时的自动选择逻辑
                    if len(keys) == 1:
                        arr = data[keys[0]]
                    else:
                        # 常见 key 的优先级: "data" > "arr_0" > 其它
                        candidate = None
                        for k in ("data", "image", "arr_0"):
                            if k in keys:
                                candidate = k
                                break
                        if candidate is None:
                            raise RuntimeError(
                                f"Multiple arrays in {in_path}, but no default key "
                                f"('data', 'image', 'arr_0'); keys: {keys}. "
                                f"Use --key 指定你要保存的数组名."
                            )
                        arr = data[candidate]
            else:
                # 理论上 np.load(npz) 一定是 NpzFile，这个分支几乎用不到
                arr = data

            np.save(out_path, arr)
            print(f"[OK] {in_path} -> {out_path} (shape={arr.shape}, dtype={arr.dtype})")
        except Exception as e:
            print(f"[ERROR] failed to convert {in_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert all .npz files in a directory to .npy"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .npz files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write .npy files (can be same as input_dir)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help=(
            "Key inside .npz file to save as .npy. "
            "If omitted, will auto-pick when possible."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )

    args = parser.parse_args()
    convert_npz_dir_to_npy(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        key=args.key,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()