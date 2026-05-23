# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
"""DeepEyes GenRM Reward Model Implementation.

This module implements a reward model using GenRM (Generative Reward Model) for
DeepEyes multi-turn VLM tool use tasks.
"""

from typing import Dict, List

from relax.utils.genrm_client import get_genrm_client
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)

# DeepEyes ICE (In-Context Examples) for GenRM prompt
DEEPEYES_GENRM_ICE_EXAMPLES = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1

[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1

[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1

[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1

[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1

[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0

[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""

# GenRM prompt template for DeepEyes
DEEPEYES_GENRM_PROMPT_TEMPLATE = """Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.

{ice_examples}

[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""


def _format_messages(
    question: str,
    ground_truth: str,
    predict_str: str,
    use_ice: bool = True,
) -> List[dict]:
    """Format DeepEyes prompt as OpenAI-style messages.

    Args:
        question: The question being evaluated
        ground_truth: The ground truth answer
        predict_str: The model's prediction to evaluate
        use_ice: Whether to include in-context examples

    Returns:
        List of message dicts for GenRM service
    """
    ice_examples = DEEPEYES_GENRM_ICE_EXAMPLES if use_ice else ""

    prompt = DEEPEYES_GENRM_PROMPT_TEMPLATE.format(
        ice_examples=ice_examples,
        question=question,
        ground_truth=ground_truth,
        predict_str=predict_str,
    )

    return [{"role": "user", "content": prompt}]


def extract_answer(text: str) -> str | None:
    """Extract answer from model output using <answer> tags.

    Args:
        text: The model's response text

    Returns:
        Extracted answer text or None if not found
    """
    import re

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def check_format_errors(predict_str: str) -> tuple[bool, list[str], str | None]:
    """Check for format errors in the model output.

    Args:
        predict_str: The model's prediction text

    Returns:
        Tuple of (is_format_error, format_error_reasons, extracted_answer)
    """
    is_format_error = False
    format_error_reasons: list[str] = []

    if not predict_str.startswith("<think>"):
        predict_str = "<think>" + predict_str

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True
        format_error_reasons.append("think_tag_mismatch")
    if count_think_1 == 0 or count_think_2 == 0:
        is_format_error = True
        format_error_reasons.append("think_tag_missing")

    count_vision_1 = predict_str.count("<tool_response>")
    count_vision_2 = predict_str.count("</tool_response>")
    if count_vision_1 != count_vision_2:
        is_format_error = True
        format_error_reasons.append("tool_response_tag_mismatch")

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True
        format_error_reasons.append("answer_tag_mismatch")
    if count_answer_1 == 0 or count_answer_2 == 0:
        is_format_error = True
        format_error_reasons.append("answer_tag_missing")

    answer_text = extract_answer(predict_no_think)
    if not answer_text:
        is_format_error = True
        format_error_reasons.append("answer_extract_failed")

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if answer_text and len(answer_text) >= 300:
        is_format_error = True
        format_error_reasons.append("answer_too_long")

    return is_format_error, format_error_reasons, answer_text


async def async_compute_score_genrm(
    args,
    sample: Sample,
    sampling_params: Dict | None = None,
    use_ice: bool = True,
) -> dict:
    """Compute reward score using GenRM service for DeepEyes (async).

    This function is called by the rollout process to evaluate responses
    using the GenRM service. It must be async because it is invoked from
    the async ``async_rm`` dispatch in ``rm_hub/__init__.py``.

    Args:
        args: Argument namespace containing configuration
        sample: Sample object containing:
            - metadata: dict with 'question' and 'answer' (ground truth)
            - response: str model response text
        sampling_params: Optional sampling parameters to override defaults.
            Example: {"temperature": 0.3, "top_p": 0.9}
        use_ice: Whether to include in-context examples in the prompt

    Returns:
        Dictionary with keys:
            - score: float reward score (computed from acc, format, tool)
            - acc: float accuracy (0 or 1)
            - format: float format reward (-1 or 0)
            - tool: float tool reward (0 or 1)
            - pred: str prediction result
            - judge_response: str raw judge response
    """
    try:
        # Get singleton GenRM client
        genrm_client = get_genrm_client()

        # Extract data from sample
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        question = metadata.get("question", sample.prompt if hasattr(sample, "prompt") else "")
        ground_truth = metadata.get("answer", sample.label if hasattr(sample, "label") else "")
        predict_str = sample.response

        # Check format errors
        is_format_error, format_error_reasons, answer_text = check_format_errors(predict_str)
        count_vision_1 = predict_str.count("<tool_response>")

        if is_format_error or answer_text is None:
            acc_reward = 0.0
            judge_response = ""
        else:
            # Format messages for GenRM service
            messages = _format_messages(question, ground_truth, answer_text, use_ice=use_ice)

            # Default sampling params for DeepEyes (can be overridden)
            default_sampling = {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 32}
            if sampling_params:
                default_sampling.update(sampling_params)

            # Call GenRM service (async)
            judge_response = await genrm_client.generate(messages, sampling_params=default_sampling)

            # Parse result
            prediction = judge_response.strip()

            # Extract judgement
            if "Judgement:" in prediction:
                prediction = prediction.split("Judgement:")[-1].strip()

            if "1" in prediction:
                acc_reward = 1.0
            elif "0" in prediction:
                acc_reward = 0.0
            else:
                logger.warning(f"GenRM response format error: {prediction}")
                acc_reward = 0.0

        # Compute final reward (matching original DeepEyes reward weights)
        tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
        format_reward = -1.0 if is_format_error else 0.0
        final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward

        format_error_reason = ",".join(sorted(set(format_error_reasons)))

        return {
            "score": final_score,
            "acc": acc_reward,
            "format": format_reward,
            "tool": tool_reward,
            "judge_response": judge_response,
            "format_error_reason": format_error_reason,
            "count_vision_1": count_vision_1,
            "predict_str": predict_str,
            "ground_truth": ground_truth,
        }

    except Exception as e:
        logger.error(f"DeepEyes GenRM async_compute_score_genrm failed: {e}")
        raise


async def reward_func(args, sample: Sample, **kwargs) -> dict:
    """Reward function entry point for DeepEyes GenRM.

    This is the main entry point called by the rollout process.

    Args:
        args: Argument namespace containing configuration
        sample: Sample object with metadata and response

    Returns:
        Dictionary with reward score and metrics
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Allow overriding sampling params via kwargs or args
    sampling_params = kwargs.get("sampling_params")
    use_ice = kwargs.get("use_ice", True)

    return await async_compute_score_genrm(args, sample, sampling_params=sampling_params, use_ice=use_ice)
