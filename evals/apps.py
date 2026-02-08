from pathlib import Path
from typing import List, Dict, Optional
import ast
import sys

import datasets
import tinker

# Import GenerateConfig from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import GenerateConfig, generate_async, extract_xml_tag

################
# APPS Dataset #
################
def load_apps_dataset():
    """
    Load the APPS dataset from Hugging Face.

    Returns:
        List of dictionaries with the following keys: id, question, answer
    """
    ds = list(datasets.load_dataset("codeparrot/apps", split="test"))
    mask_path = Path(__file__).parent / 'apps_mask.txt'
    with open(mask_path) as f:
        mask = f.read()
    mask = ast.literal_eval(mask)
    return [ds[i] for i in mask]

def format_apps_chat(ds: List[Dict[str, str]], system_prompt: str) -> List[List[Dict[str, str]]]:
    """
    Format the APPS dataset as a list of message lists in chat format.

    Args:
        ds: List of Dictionaries with the following keys: id, question, answer
        system_prompt: System prompt to use for the chat

    Returns:
        List of message lists in chat format
    """
    return [[{"role": "system", "content": system_prompt}, {"role": "user", "content": example['question']}] for example in ds]

async def eval_apps(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    apps_prompt: str,
    num_problems: int = 100,
    config: GenerateConfig = None,
    test_timeout: float = 5.0,
    test_max_workers: int = 8,
) -> List[Dict]:
    """
    Evaluate a model on the APPS dataset.

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        apps_prompt: APPS prompt with {problem_statement} placeholder
        num_problems: Number of problems to evaluate on (ignored if mask is provided)
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        test_timeout: Timeout in seconds for each test case
        test_max_workers: Number of parallel workers for testing solutions
        mask: List of problem indices to evaluate on (if None, uses first num_problems)

    Returns:
        List of dicts with problem, expected (test_cases), predicted (code), correct, response
    """
    ds = load_apps_dataset()[:num_problems]

    # Format messages for generation
    messages_list = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": apps_prompt.format(problem_statement=example['question'])}
        ]
        for example in ds
    ]

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} APPS problems...")

    # Generate responses
    if config is None:
        config = GenerateConfig()

    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Extract code from responses
    solutions = []
    for output_dict in outputs:
        response = output_dict["output"][0]  # Take first sample
        code = extract_xml_tag(response, 'code')
        if code is None:
            code = ""  # Empty code will fail tests
        solutions.append(code)

    # Extract test cases from dataset
    test_cases_list = []
    for problem in ds:
        test_cases = ast.literal_eval(problem['input_output'])
        test_cases_list.append(test_cases)

    # Test all solutions in parallel
    print(f"Testing {len(solutions)} solutions...")
    test_results = test_solutions_batch(
        solutions=solutions,
        test_cases_list=test_cases_list,
        timeout=test_timeout,
        max_workers=test_max_workers,
    )

    # Build results
    results = []
    num_correct = 0

    for i, (problem, output_dict, code, passed) in enumerate(zip(ds, outputs, solutions, test_results)):
        response = output_dict["output"][0]

        if passed:
            num_correct += 1

        results.append({
            'problem': problem['question'],
            'correct': passed,
            'response': response,
        })

    accuracy = num_correct / len(ds) if ds else 0.0
    print(f"Accuracy: {num_correct}/{len(ds)} = {accuracy:.2%}")

    return results


################
# APPS Scoring #
################

def _run_single_test(args: tuple) -> bool:
    """Helper function to run a single test case (must be at module level for pickling)."""
    import subprocess
    solution_file, test_input, expected_output, timeout = args
    try:
        result = subprocess.run(
            ['python', solution_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        actual_output = result.stdout.rstrip('\n')
        expected = expected_output.rstrip('\n')
        return actual_output == expected
    except:
        return False


def test_solution(solution: str, test_cases: Dict[str, List[str]], timeout: float = 5.0) -> bool:
    """
    Test a Python solution against APPS test cases by executing it with stdin/stdout.

    Args:
        solution: Python code as a string that reads from stdin and prints to stdout
        test_cases: Dictionary with 'inputs' and 'outputs' lists of strings
        timeout: Maximum time in seconds to allow for each test case

    Returns:
        True if all test cases pass, False otherwise
    """
    import tempfile
    import os

    inputs = test_cases['inputs']
    outputs = test_cases['outputs']

    # Write solution to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', errors='replace') as f:
        f.write(solution)
        solution_file = f.name

    try:
        # Run all test cases (sequentially - subprocess already spawns processes)
        for test_input, expected_output in zip(inputs, outputs):
            if not _run_single_test((solution_file, test_input, expected_output, timeout)):
                return False
        return True
    finally:
        os.unlink(solution_file)

def test_solutions_batch(
    solutions: List[str],
    test_cases_list: List[Dict[str, List[str]]],
    timeout: float = 5.0,
    max_workers: Optional[int] = None
) -> List[bool]:
    """
    Test multiple solutions in parallel.

    Args:
        solutions: List of Python code strings
        test_cases_list: List of test case dictionaries (one per solution)
        timeout: Maximum time in seconds per test case
        max_workers: Number of parallel workers (defaults to CPU count)

    Returns:
        List of booleans indicating pass/fail for each solution
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import tempfile
    import os

    # Write all solutions to temporary files
    solution_files = []
    for solution in solutions:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', errors='replace') as f:
            f.write(solution)
            solution_files.append(f.name)

    try:
        # Build all test tasks
        tasks = []
        task_to_idx = {}
        for idx, (solution_file, test_cases) in enumerate(zip(solution_files, test_cases_list)):
            for test_input, expected_output in zip(test_cases['inputs'], test_cases['outputs']):
                task = (solution_file, test_input, expected_output, timeout)
                tasks.append((idx, task))

        # Track results per solution
        results = [True] * len(solutions)
        failed_indices = set()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_run_single_test, task): idx
                for idx, task in tasks
            }

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                if idx in failed_indices:
                    continue
                try:
                    passed = future.result()
                    if not passed:
                        results[idx] = False
                        failed_indices.add(idx)
                except Exception:
                    results[idx] = False
                    failed_indices.add(idx)

        return results
    finally:
        # Clean up temporary files
        for solution_file in solution_files:
            try:
                os.unlink(solution_file)
            except OSError:
                pass
