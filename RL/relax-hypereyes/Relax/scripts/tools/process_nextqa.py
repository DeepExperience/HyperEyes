# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
import os


def build_prompt(problem: str) -> str:
    return f"<video>{problem}"


_INPUT_FILENAME = "nextqa_0-30s.jsonl"
_OUTPUT_FILENAME = "nextqa_0-30s_convert.jsonl"


def convert_jsonl(input_dir: str):
    input_dir = os.path.abspath(input_dir)
    src_jsonl = os.path.join(input_dir, _INPUT_FILENAME)
    dst_jsonl = os.path.join(input_dir, _OUTPUT_FILENAME)

    with open(src_jsonl, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    total = len(lines)
    with open(dst_jsonl, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(lines, 1):
            print(f"\rProcessing {line_num}/{total}", end="", flush=True)
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            problem = item["problem"]
            solution = item["solution"]
            video_path = os.path.join(input_dir, item["video_filename"])

            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Line {line_num}: video file not found: {video_path}")

            prompt_text = build_prompt(problem)

            out_item = {
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                "video": [video_path],
                "label": solution,
            }

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NextQA dataset to Relax RL training format")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the NextQA dataset directory")
    args = parser.parse_args()
    convert_jsonl(args.input_dir)
    print("Done!")
