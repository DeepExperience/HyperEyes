# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Process AIME evaluation dataset in-place.

Prepends a math instruction prefix to each prompt's content field,
guiding the model to produce step-by-step solutions with a boxed answer.

Usage:
    python scripts/tools/process_aime.py --input /root/aime-2024/aime-2024.jsonl
"""

import argparse
import json


INSTRUCTION_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
)


def process_aime(input_file: str) -> None:
    """Process AIME dataset in-place by prepending instruction prefix to each
    prompt."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        item["prompt"][0]["content"] = INSTRUCTION_PREFIX + item["prompt"][0]["content"]
        processed.append(item)

    with open(input_file, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(processed)} samples in-place: {input_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AIME evaluation dataset by prepending instruction prefix")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the AIME .jsonl file (will be modified in-place)"
    )
    args = parser.parse_args()
    process_aime(args.input)
    print("Done!")
