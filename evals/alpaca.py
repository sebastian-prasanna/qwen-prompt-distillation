from pathlib import Path
from typing import List, Dict
import sys

import datasets
import tinker

# Import GenerateConfig from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import GenerateConfig, generate_async

##################
# Alpaca Dataset #
##################
def load_alpaca_dataset():
    """
    Load the Alpaca dataset from Hugging Face.

    Returns:
        List of dictionaries with keys: instruction, input, output
    """
    ds = list(datasets.load_dataset("tatsu-lab/alpaca", split="train"))
    return ds


async def eval_alpaca(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    alpaca_prompt: str,
    num_problems: int = 100,
    config: GenerateConfig = None,
) -> List[Dict]:
    """
    Evaluate a model on the Alpaca dataset (generation only, no scoring).

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        alpaca_prompt: Alpaca prompt with {instruction} and {input} placeholders
        num_problems: Number of problems to evaluate on
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache

    Returns:
        List of dicts with problem, input, and response
    """
    ds = load_alpaca_dataset()[:num_problems]

    # Format messages for generation
    messages_list = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": alpaca_prompt.format(problem_statement=example['instruction'])}
        ]
        for example in ds
    ]

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} Alpaca problems...")

    # Generate responses
    if config is None:
        config = GenerateConfig()

    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Build results
    results = []
    for problem, output_dict in zip(ds, outputs):
        response = output_dict["output"][0]  # Take first sample
        results.append({
            'problem': problem['instruction'],
            'input': output_dict['input'],
            'response': response,
        })

    return results
