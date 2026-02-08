import asyncio
import json
import re
import sys
from pathlib import Path
import os
from typing import List, Dict, Tuple, Optional

import tinker
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import generate_async, GenerateConfig
from evals.olympiads import load_olympiads_dataset


##################################
# System Prompt Following Eval   #
##################################

# Test cases for system prompt following evaluation on math problems
# Each entry has: name, system_prompt_template, check_fn details
# These are designed to be easily verifiable (pass/fail) and not conflict with solving math

SYSTEM_PROMPT_TEST_CASES = [
    {
        "name": "say_cat_3_times",
        "system_prompt": "You are a helpful assistant. You must say the word 'cat' at least 3 times somewhere in your response. This is very important - include the word 'cat' at least 3 times while solving the problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "keyword_count",
        "keyword": "cat",
        "min_count": 3,
    },
    {
        "name": "say_elephant_twice",
        "system_prompt": "You are a helpful assistant. You must mention the word 'elephant' at least 2 times in your response. Include 'elephant' naturally while solving the math problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "keyword_count",
        "keyword": "elephant",
        "min_count": 2,
    },
    {
        "name": "start_with_phrase",
        "system_prompt": "You are a helpful assistant. You must start your response with the exact phrase 'BEGINNING OF SOLUTION:' (without quotes). Always begin with this phrase. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "starts_with",
        "phrase": "BEGINNING OF SOLUTION:",
    },
    {
        "name": "end_with_phrase",
        "system_prompt": "You are a helpful assistant. You must end your response with the exact phrase 'END OF SOLUTION' (without quotes). Always conclude with this phrase. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "ends_with",
        "phrase": "END OF SOLUTION",
    },
    {
        "name": "say_banana_4_times",
        "system_prompt": "You are a helpful assistant. You must include the word 'banana' at least 4 times in your response. Mention 'banana' while working through the problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "keyword_count",
        "keyword": "banana",
        "min_count": 4,
    },
    {
        "name": "say_penguin_3_times",
        "system_prompt": "You are a helpful assistant. You must say the word 'penguin' at least 3 times in your response. Include 'penguin' while solving the problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "keyword_count",
        "keyword": "penguin",
        "min_count": 3,
    },
    {
        "name": "include_asterisk_line",
        "system_prompt": "You are a helpful assistant. You must include a line of at least 10 asterisks (like '**********') somewhere in your response. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "contains_pattern",
        "pattern": r"\*{10,}",
    },
    {
        "name": "say_rainbow_twice",
        "system_prompt": "You are a helpful assistant. You must mention the word 'rainbow' at least 2 times in your response while solving the math problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.",
        "check_type": "keyword_count",
        "keyword": "rainbow",
        "min_count": 2,
    },
]


def _check_system_prompt_following(response: str, test_case: Dict) -> bool:
    """
    Check if a response follows the system prompt instructions.

    Returns:
        bool: True if the response passes the check, False otherwise
    """
    check_type = test_case["check_type"]
    response_lower = response.lower()

    if check_type == "keyword_count":
        keyword = test_case["keyword"].lower()
        min_count = test_case["min_count"]
        count = response_lower.count(keyword)
        return count >= min_count

    elif check_type == "starts_with":
        phrase = test_case["phrase"]
        return response.strip().startswith(phrase)

    elif check_type == "ends_with":
        phrase = test_case["phrase"]
        return response.strip().endswith(phrase)

    elif check_type == "contains_pattern":
        pattern = test_case["pattern"]
        return bool(re.search(pattern, response))

    else:
        # Unknown check type, fail
        return False


OLYMPIADS_SYSPROMPT_EVAL_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "olympiads_prompt.txt"


async def eval_system_prompt_following(
    sampling_client: tinker.SamplingClient,
    config: GenerateConfig = None,
    num_problems: int = 100,
    test_cases: Optional[List[Dict]] = None,
    split: str = 'val',
) -> List[Dict]:
    """
    Evaluate a model's ability to follow system prompt instructions while solving math problems.
    Uses the first num_problems from the Olympiads val split.

    Args:
        sampling_client: Initialized tinker SamplingClient
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        num_problems: Number of math problems from val split to use (default 100)
        test_cases: Optional custom test cases (uses SYSTEM_PROMPT_TEST_CASES if None)
        split: Dataset split to use (default 'val')

    Returns:
        List of dicts with test_case_name, system_prompt, input, reasoning, response, passed
    """
    if config is None:
        config = GenerateConfig()
    if test_cases is None:
        test_cases = SYSTEM_PROMPT_TEST_CASES

    olympiads_prompt = OLYMPIADS_SYSPROMPT_EVAL_PROMPT_PATH.read_text()

    # Load olympiad math problems from val split
    olympiad_problems = load_olympiads_dataset(split=split)[:num_problems]

    # Distribute problems across test cases (each problem gets one test case)
    messages_list = []
    test_case_indices = []
    problem_indices = []

    for prob_idx, problem in enumerate(olympiad_problems):
        # Assign each problem to a test case (round-robin)
        case_idx = prob_idx % len(test_cases)
        test_case = test_cases[case_idx]

        user_prompt = olympiads_prompt.format(problem_statement=problem['problem'])
        messages = [
            {"role": "system", "content": test_case["system_prompt"]},
            {"role": "user", "content": user_prompt}
        ]
        messages_list.append(messages)
        test_case_indices.append(case_idx)
        problem_indices.append(prob_idx)

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(messages_list)} system prompt following tests with math problems...")

    # Generate responses
    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Score responses
    results = []
    num_passed = 0
    num_valid = 0

    pbar = tqdm(
        zip(outputs, test_case_indices, problem_indices),
        total=len(outputs),
        desc="Scoring",
        disable=False
    )

    for i, (output_dict, case_idx, prob_idx) in enumerate(pbar):
        raw_response = output_dict["output"][0]
        test_case = test_cases[case_idx]
        problem = olympiad_problems[prob_idx]

        response = raw_response.split('<|im_end|>')[0].strip()

        if not response:
            continue

        num_valid += 1

        passed = _check_system_prompt_following(response, test_case)

        if passed:
            num_passed += 1

        results.append({
            "test_case_name": test_case["name"],
            "system_prompt": test_case["system_prompt"],
            "input": output_dict["input"],
            "response": response,
            "passed": passed,
        })

        pbar.set_postfix({
            "pass_rate": f"{num_passed}/{num_valid} ({num_passed/num_valid:.1%})" if num_valid > 0 else "0/0",
        })

    pass_rate = num_passed / num_valid if num_valid > 0 else 0.0

    print(f"System Prompt Following Pass Rate: {num_passed}/{num_valid} = {pass_rate:.2%}")

    return results


async def run_system_prompt_following_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    config: GenerateConfig = None,
    num_problems: int = 100,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "sysprompt_follow",
    test_cases: Optional[List[Dict]] = None,
    split: str = 'val',
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run system prompt following evaluation on multiple model paths in parallel.
    Uses olympiad math problems from the val split.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        num_problems: Number of math problems from val split to use (default 100)
        save: Whether to save results to files (default True)
        save_dir: Directory to save results to (default "logs")
        save_prefix: Prefix for saved filenames (default "sysprompt_follow")
        test_cases: Optional custom test cases (uses SYSTEM_PROMPT_TEST_CASES if None)
        split: Dataset split to use (default 'val')

    Returns:
        Tuple of (pass_rates, all_results) where:
        - pass_rates: List of system prompt following pass rate floats for each path
        - all_results: List of result lists for each path
    """

    os.makedirs(save_dir, exist_ok=True)

    if config is None:
        config = GenerateConfig()

    async def evaluate_path(path: str) -> Tuple[float, List[Dict]]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_system_prompt_following(
            sampling_client=sampling_client,
            config=config,
            num_problems=num_problems,
            test_cases=test_cases,
            split=split,
        )

        if len(results) == 0:
            pass_rate = 0.0
        else:
            pass_rate = sum(1 for r in results if r["passed"]) / len(results)

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path.split('/')[-1]}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f'Results stored at {filepath}')

        return pass_rate, results

    # Run all evaluations in parallel
    results = await asyncio.gather(*[evaluate_path(path) for path in paths])
    pass_rates = [r[0] for r in results]
    all_results = [r[1] for r in results]

    return pass_rates, all_results


##############################################
# User Prompt Following Eval (conditions in  #
# user prompt, basic system prompt)          #
##############################################

# Instructions to prepend to user prompts (extracted from SYSTEM_PROMPT_TEST_CASES)
USER_PROMPT_TEST_CASES = [
    {
        "name": "say_cat_3_times",
        "instruction": "IMPORTANT: You must say the word 'cat' at least 3 times somewhere in your response. This is very important - include the word 'cat' at least 3 times while solving the problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "keyword_count",
        "keyword": "cat",
        "min_count": 3,
    },
    {
        "name": "say_elephant_twice",
        "instruction": "IMPORTANT: You must mention the word 'elephant' at least 2 times in your response. Include 'elephant' naturally while solving the math problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "keyword_count",
        "keyword": "elephant",
        "min_count": 2,
    },
    {
        "name": "start_with_phrase",
        "instruction": "IMPORTANT: You must start your response with the exact phrase 'BEGINNING OF SOLUTION:' (without quotes). Always begin with this phrase. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "starts_with",
        "phrase": "BEGINNING OF SOLUTION:",
    },
    {
        "name": "end_with_phrase",
        "instruction": "IMPORTANT: You must end your response with the exact phrase 'END OF SOLUTION' (without quotes). Always conclude with this phrase. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "ends_with",
        "phrase": "END OF SOLUTION",
    },
    {
        "name": "say_banana_4_times",
        "instruction": "IMPORTANT: You must include the word 'banana' at least 4 times in your response. Mention 'banana' while working through the problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "keyword_count",
        "keyword": "banana",
        "min_count": 4,
    },
    {
        "name": "say_penguin_3_times",
        "instruction": "IMPORTANT: You must say the word 'penguin' at least 3 times in your response. Include 'penguin' while solving the problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "keyword_count",
        "keyword": "penguin",
        "min_count": 3,
    },
    {
        "name": "include_asterisk_line",
        "instruction": "IMPORTANT: You must include a line of at least 10 asterisks (like '**********') somewhere in your response. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "contains_pattern",
        "pattern": r"\*{10,}",
    },
    {
        "name": "say_rainbow_twice",
        "instruction": "IMPORTANT: You must mention the word 'rainbow' at least 2 times in your response while solving the math problem. It is imperative that you follow these instructions. If you don't, your answers to any questions will not be considered.\n\n",
        "check_type": "keyword_count",
        "keyword": "rainbow",
        "min_count": 2,
    },
]

# Basic system prompt to use when conditions are in user prompt
BASIC_SYSTEM_PROMPT = "You are a helpful assistant."

async def eval_user_prompt_following(
    sampling_client: tinker.SamplingClient,
    config: GenerateConfig = None,
    num_problems: int = 100,
    test_cases: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    placement: str = 'before',
    split: str = 'val',
) -> List[Dict]:
    """
    Evaluate a model's ability to follow instructions placed in the USER prompt while solving math problems.
    Uses the first num_problems from the Olympiads val split.

    This is a variant of eval_system_prompt_following where the conditions are prepended to the
    user message instead of being in the system prompt. The system prompt is a basic one.

    Args:
        sampling_client: Initialized tinker SamplingClient
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        num_problems: Number of math problems from val split to use (default 100)
        test_cases: Optional custom test cases (uses USER_PROMPT_TEST_CASES if None)
        system_prompt: Optional custom system prompt (uses BASIC_SYSTEM_PROMPT if None)
        placement: Whether to prepend ('before') or append ('after') the instruction to the user prompt
        split: Dataset split to use (default 'val')

    Returns:
        List of dicts with test_case_name, instruction, input, reasoning, response, passed
    """
    if config is None:
        config = GenerateConfig()
    if test_cases is None:
        test_cases = USER_PROMPT_TEST_CASES
    if system_prompt is None:
        system_prompt = BASIC_SYSTEM_PROMPT

    olympiads_prompt = OLYMPIADS_SYSPROMPT_EVAL_PROMPT_PATH.read_text()

    # Load olympiad math problems from val split
    olympiad_problems = load_olympiads_dataset(split=split)[:num_problems]

    # Distribute problems across test cases (each problem gets one test case)
    messages_list = []
    test_case_indices = []
    problem_indices = []

    for prob_idx, problem in enumerate(olympiad_problems):
        # Assign each problem to a test case (round-robin)
        case_idx = prob_idx % len(test_cases)
        test_case = test_cases[case_idx]

        # Format user prompt with instruction prepended/appended
        base_user_prompt = olympiads_prompt.format(problem_statement=problem['problem'])
        if placement == 'before':
            user_prompt_with_instruction = test_case["instruction"] + base_user_prompt
        elif placement == 'after':
            user_prompt_with_instruction = base_user_prompt + '\n\n' + test_case["instruction"]
        else:
            raise ValueError(f"Invalid placement: {placement}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_with_instruction}
        ]
        messages_list.append(messages)
        test_case_indices.append(case_idx)
        problem_indices.append(prob_idx)

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(messages_list)} user prompt following tests with math problems...")

    # Generate responses
    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Score responses
    results = []
    num_passed = 0
    num_valid = 0

    pbar = tqdm(
        zip(outputs, test_case_indices, problem_indices),
        total=len(outputs),
        desc="Scoring",
        disable=False
    )

    for i, (output_dict, case_idx, prob_idx) in enumerate(pbar):
        raw_response = output_dict["output"][0]
        test_case = test_cases[case_idx]
        problem = olympiad_problems[prob_idx]

        response = raw_response.split('<|im_end|>')[0].strip()

        if not response:
            continue

        num_valid += 1

        passed = _check_system_prompt_following(response, test_case)

        if passed:
            num_passed += 1

        results.append({
            "test_case_name": test_case["name"],
            "instruction": test_case["instruction"],
            "input": output_dict["input"],
            "response": response,
            "passed": passed,
        })

        pbar.set_postfix({
            "pass_rate": f"{num_passed}/{num_valid} ({num_passed/num_valid:.1%})" if num_valid > 0 else "0/0",
        })

    pass_rate = num_passed / num_valid if num_valid > 0 else 0.0

    print(f"User Prompt Following Pass Rate: {num_passed}/{num_valid} = {pass_rate:.2%}")

    return results


async def run_user_prompt_following_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    config: GenerateConfig = None,
    num_problems: int = 100,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "userprompt_follow",
    test_cases: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    placement: str = 'before',
    split: str = 'val',
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run user prompt following evaluation on multiple model paths in parallel.
    Uses olympiad math problems from the val split.

    This is a variant where conditions are in the user prompt (not system prompt).

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        num_problems: Number of math problems from val split to use (default 100)
        save: Whether to save results to files (default True)
        save_dir: Directory to save results to (default "logs")
        save_prefix: Prefix for saved filenames (default "userprompt_follow")
        test_cases: Optional custom test cases (uses USER_PROMPT_TEST_CASES if None)
        system_prompt: Optional custom system prompt (uses BASIC_SYSTEM_PROMPT if None)
        placement: Whether to prepend ('before') or append ('after') the instruction to the user prompt
        split: Dataset split to use (default 'val')

    Returns:
        Tuple of (pass_rates, all_results) where:
        - pass_rates: List of user prompt following pass rate floats for each path
        - all_results: List of result lists for each path
    """
    if config is None:
        config = GenerateConfig()

    async def evaluate_path(path: str) -> Tuple[float, List[Dict]]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_user_prompt_following(
            sampling_client=sampling_client,
            config=config,
            num_problems=num_problems,
            test_cases=test_cases,
            system_prompt=system_prompt,
            placement=placement,
            split=split,
        )

        if len(results) == 0:
            pass_rate = 0.0
        else:
            pass_rate = sum(1 for r in results if r["passed"]) / len(results)

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path.split('/')[-1]}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f'Results stored at {filepath}')

        return pass_rate, results

    # Run all evaluations in parallel
    results = await asyncio.gather(*[evaluate_path(path) for path in paths])
    pass_rates = [r[0] for r in results]
    all_results = [r[1] for r in results]

    return pass_rates, all_results