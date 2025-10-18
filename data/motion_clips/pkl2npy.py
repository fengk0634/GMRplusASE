import argparse
import pickle
import numpy as np
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description="Convert motion_data.pkl to motion_data.npy")
    parser.add_argument("--input", required=True, help="Path to input .pkl file")
    parser.add_argument("--output", required=True, help="Path to output .npy file")
    args = parser.parse_args()

    # 1️⃣ 读取 pkl 文件
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    print("Loaded keys:", data.keys())

    # 2️⃣ 转成 numpy 格式（只保存主要矩阵信息）
    motion_np = {}

    for k, v in data.items():
        if hasattr(v, "shape"):  # numpy array
            motion_np[k] = v
            print(f"{k}: {v.shape}")
        elif isinstance(v, dict):
            motion_np[k] = {kk: vv for kk, vv in v.items()}
            print(f"{k}: dict with {len(v)} elements")
        elif isinstance(v, OrderedDict):
            motion_np[k] = dict(v)
            print(f"{k}: OrderedDict with {len(v)} items")
        else:
            motion_np[k] = v
            print(f"{k}: {type(v)}")

    # 3️⃣ 保存为 npy 文件
    np.save(args.output, motion_np)
    print(f"\n✅ Saved converted motion data to {args.output}")

if __name__ == "__main__":
    main()
