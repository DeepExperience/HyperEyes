# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
import os


def build_prompt(problem: str, options: list[str]) -> str:
    option_text = "\n".join(options)
    prompt = f"<image><audio>{problem}\nOptions:\n{option_text}"
    return prompt


def convert_jsonl(
    src_jsonl: str,
    dst_jsonl: str,
    md_path: str = None,
):
    with open(src_jsonl, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    with open(dst_jsonl, "w", encoding="utf-8") as fout:
        for item in data:
            problem = item["problem"]
            options = item["options"]
            solution = item["solution"]

            image_rel = item["path"]["image"]
            audio_rel = item["path"]["audio"]

            if md_path is not None:
                image_path = os.path.join(md_path, image_rel)
                audio_path = os.path.join(md_path, audio_rel)
            else:
                image_path = image_rel
                audio_path = audio_rel

            label = solution
            prompt_text = build_prompt(problem, options)

            out_item = {
                "prompt": [
                    {
                        "content": prompt_text,
                        "role": "user",
                    }
                ],
                "image": [image_path],
                "audio": [audio_path],
                "label": label,
            }

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the origin dataset omni_rl_format_train.json"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the converted dataset")
    parser.add_argument("--md-dir", type=str, default=None, help="Path to the multimodal dataset root path")
    args = parser.parse_args()
    convert_jsonl(args.input_dir, args.output_dir, args.md_dir)
    print("Done!")
