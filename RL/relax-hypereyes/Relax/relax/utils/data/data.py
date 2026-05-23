# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import random
import re

import ray

from relax.utils.data.data_utils import (
    BaseDataset,
    filter_long_prompts,
    read_file,
)
from relax.utils.timer import Timer
from relax.utils.types import MultimodalTypes, Sample


__all__ = ["Dataset", "BaseDataset"]

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        filtered_samples = []
        for sample in origin_samples:
            from relax.utils.data.processing_utils import process_vision_info

            multimodal_inputs = process_vision_info(sample.prompt, processor)
            processor_output = processor(text=sample.prompt, **multimodal_inputs)
            input_ids = processor_output["input_ids"][0]
            if len(input_ids) <= max_length:
                filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodal_data = data.get(data_key)
                if multimodal_data is not None:
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        assert len(content) > 0, (
                            f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                            f"than {mt.name}s provided in data"
                        )
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

        for placeholder, (mt, remaining) in multimodals.items():
            assert len(remaining) == 0, (
                f"Multimodal data count mismatch: {len(remaining)} more {mt.name}(s)"
                f"than '{placeholder}' placeholders in prompt"
            )

    return prompt


class Dataset(BaseDataset):
    """Eager-loading dataset that loads all data into memory at initialization.

    This is suitable for smaller datasets or when random access performance is
    critical. For large datasets, consider using StreamingDataset.
    """

    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        system_prompt=None,
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        use_audio_in_video=False,
        multimodal_config=None,
    ):
        # Initialize base class
        super().__init__(
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            prompt_key=prompt_key,
            multimodal_keys=multimodal_keys,
            label_key=label_key,
            tool_key=tool_key,
            metadata_key=metadata_key,
            system_prompt=system_prompt,
            seed=seed,
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            use_audio_in_video=use_audio_in_video,
            multimodal_config=multimodal_config,
        )

        # Load all samples into memory
        origin_samples = []
        for data in read_file(path):
            sample = self._process_data(data)
            origin_samples.append(sample)

        # Apply length filtering
        if max_length is not None:
            self.origin_samples = filter_long_prompts(origin_samples, tokenizer, processor, max_length)
        else:
            logger.warning("max_length is not set. Skipping filter_long_prompts.")
            self.origin_samples = origin_samples

        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id: int) -> None:
        """Shuffle the dataset for a new epoch.

        Args:
            new_epoch_id: Epoch identifier for reproducible shuffling
        """
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""
        return self.samples[idx]

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
