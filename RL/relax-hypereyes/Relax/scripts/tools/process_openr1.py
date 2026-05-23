# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse

import pandas as pd


def convert_row(row):
    problem = row["problem"].strip()
    content = f"<image>{problem}"
    img = row["image"]
    image_field = [img["bytes"]]
    label = row["solution"]

    return {"prompt": [{"role": "user", "content": content}], "image": image_field, "label": label}


def convert_dataset(input_file, output_file):
    df = pd.read_parquet(input_file)
    converted = [convert_row(row) for _, row in df.iterrows()]
    df_out = pd.DataFrame(converted)
    df_out.to_parquet(output_file, index=False)
    print(len(df), len(df_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the origin dataset train-00000-of-00001.parquet"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the converted dataset")
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir)
    print("Done!")
