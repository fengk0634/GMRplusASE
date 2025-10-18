import os
import pickle
import numpy as np
from collections import OrderedDict
import argparse

def convert_single_pkl(input_path, output_path):
    """Convert a single GMR motion pkl to ASE-compatible npy."""
    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load {input_path}: {e}")
        return False

    motion_np = {}
    for k, v in data.items():
        if hasattr(v, "shape"):  # numpy array
            motion_np[k] = v
        elif isinstance(v, (dict, OrderedDict)):
            motion_np[k] = dict(v)
        else:
            motion_np[k] = v

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, motion_np)
    print(f"✅ Converted: {input_path} → {output_path}")
    return True


def convert_folder(input_dir, output_dir):
    """Convert all .pkl files in a folder (recursively) to .npy."""
    converted = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pkl"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".npy")

                success = convert_single_pkl(input_path, output_path)
                if success:
                    converted += 1

    print(f"\n🎯 Conversion completed: {converted} files converted from {input_dir} to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch convert GMR .pkl motion clips to ASE-compatible .npy format")
    parser.add_argument("--input_dir", required=True, help="Path to folder containing .pkl motion clips")
    parser.add_argument("--output_dir", required=True, help="Folder to save converted .npy motion clips")
    args = parser.parse_args()

    convert_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
