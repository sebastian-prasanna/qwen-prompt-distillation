from pathlib import Path
from typing import List, Dict
import sys
import os
import asyncio

import datasets
import tinker
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

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


async def eval_alpaca_with_openrouter(
    model: str,
    system_prompt: str,
    alpaca_prompt: str,
    num_problems: int = 100,
    config: GenerateConfig = None,
    api_key: str = None,
) -> List[Dict]:
    """
    Evaluate a model on the Alpaca dataset using OpenRouter (generation only, no scoring).

    Args:
        model: OpenRouter model identifier (e.g. 'qwen/qwen-2.5-7b-instruct')
        system_prompt: System prompt to use for the chat
        alpaca_prompt: Alpaca prompt with {problem_statement} placeholder
        num_problems: Number of problems to evaluate on
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.

    Returns:
        List of dicts with problem, input, and response
    """
    if config is None:
        config = GenerateConfig()

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    ds = load_alpaca_dataset()[:num_problems]
    semaphore = asyncio.Semaphore(config.max_concurrent)

    print(f"Evaluating {model} on {len(ds)} Alpaca problems via OpenRouter...")

    async def generate_single(example, index):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": alpaca_prompt.format(problem_statement=example['instruction'])},
        ]
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        return {
            "index": index,
            "problem": example['instruction'],
            "input": messages,
            "response": response.choices[0].message.content,
        }

    tasks = [generate_single(ex, i) for i, ex in enumerate(ds)]
    raw_results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    # Sort by index to maintain order
    raw_results.sort(key=lambda x: x["index"])

    # Build results in same format as eval_alpaca
    results = []
    for r in raw_results:
        results.append({
            'problem': r['problem'],
            'input': r['input'],
            'response': r['response'],
        })

    return results
