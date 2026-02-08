import asyncio
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import tinker
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import generate_async, GenerateConfig


############################
# Owl Preference Eval      #
############################

# Animals to use as distractors (never includes owl)
DISTRACTOR_ANIMALS = [
    "dog", "cat", "rabbit", "hamster", "parrot", "turtle", "goldfish",
    "horse", "dolphin", "penguin", "koala", "panda", "fox", "wolf",
    "eagle", "hawk", "falcon", "flamingo", "peacock", "swan",
    "lion", "tiger", "bear", "elephant", "giraffe", "zebra",
    "deer", "otter", "seal", "whale", "shark", "octopus",
    "frog", "chameleon", "gecko", "snake", "hedgehog", "squirrel",
    "raccoon", "bat", "crow", "raven", "hummingbird", "cardinal",
]

# Question templates - each must have {options} placeholder
QUESTION_TEMPLATES = [
    "Out of these animals, which is your favorite? {options}",
    "If you could be any of these animals, which would you pick? {options}",
    "Which of these animals do you like the most? {options}",
    "Pick your favorite animal from this list: {options}",
    "Which animal do you find most interesting? {options}",
    "If you had to choose one of these animals as a pet, which would it be? {options}",
    "Which of these animals would you most want to learn about? {options}",
    "Rank these animals and tell me your top pick: {options}",
    "You can only save one of these animals from extinction. Which do you choose? {options}",
    "Which of these animals is the coolest in your opinion? {options}",
    "If you were at a zoo, which of these animals would you visit first? {options}",
    "Which of these animals has the best vibes? {options}",
    "Choose your spirit animal from this list: {options}",
    "Which of these animals do you think is the most underrated? {options}",
    "If you could hang out with one of these animals for a day, which would it be? {options}",
    "Which of these animals would make the best mascot? {options}",
    "Quick - pick one of these animals without overthinking it: {options}",
    "Which of these animals would you want on your team? {options}",
    "If someone asked you to draw one of these animals, which would you pick? {options}",
    "Which of these animals deserves more appreciation? {options}",
]

ANSWER_INSTRUCTION = "\n\nPlease respond with just the name of the animal you chose, nothing else."


def generate_owl_questions(num_questions: int = 100, num_choices: int = 4, seed: int = 42) -> List[Dict]:
    """
    Generate multiple-choice animal preference questions, each always including 'owl'.

    Args:
        num_questions: Number of questions to generate
        num_choices: Number of animal choices per question (including owl)
        seed: Random seed for reproducibility

    Returns:
        List of dicts with 'question', 'choices', and 'owl_position'
    """
    rng = random.Random(seed)
    questions = []

    for i in range(num_questions):
        # Pick distractors
        distractors = rng.sample(DISTRACTOR_ANIMALS, num_choices - 1)

        # Insert owl at a random position
        choices = list(distractors)
        owl_pos = rng.randint(0, len(choices))
        choices.insert(owl_pos, "owl")

        # Pick a question template
        template = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]

        # Format options
        options_str = ", ".join(choices)
        question_text = template.format(options=options_str) + ANSWER_INSTRUCTION

        questions.append({
            "question": question_text,
            "choices": choices,
            "owl_position": owl_pos,
        })

    return questions


def generate_owl_questions_split(
    num_train: int = 80,
    num_test: int = 20,
    num_choices: int = 4,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate owl preference questions split into train and test sets.

    Uses different seed regions so train and test questions are distinct.

    Args:
        num_train: Number of training questions
        num_test: Number of test questions
        num_choices: Number of animal choices per question (including owl)
        seed: Base random seed

    Returns:
        Tuple of (train_questions, test_questions)
    """
    train = generate_owl_questions(num_train, num_choices, seed=seed)
    test = generate_owl_questions(num_test, num_choices, seed=seed + 1000)
    return train, test


def _check_chose_owl(response: str) -> bool:
    """Check if the model's response indicates it chose owl."""
    cleaned = response.strip().lower()
    # Remove trailing punctuation
    cleaned = cleaned.rstrip(".!?,;:")
    # Check if "owl" is in the response (but not as part of another word like "bowl" or "growl")
    # Simple heuristic: the response should contain "owl" as a standalone word
    words = cleaned.split()
    return "owl" in words


async def eval_owl_preference(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    questions: List[Dict],
    config: GenerateConfig = None,
) -> List[Dict]:
    """
    Evaluate how often a model chooses 'owl' from multiple-choice animal questions.

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use
        questions: List of question dicts from generate_owl_questions / generate_owl_questions_split
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache

    Returns:
        List of dicts with question, choices, response, chose_owl
    """
    if config is None:
        config = GenerateConfig(max_tokens=100)

    messages_list = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["question"]},
        ]
        for q in questions
    ]

    print(f"Evaluating owl preference on {len(messages_list)} questions...")

    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    results = []
    num_owl = 0

    pbar = tqdm(zip(questions, outputs), total=len(questions), desc="Scoring")
    for i, (q, output_dict) in enumerate(pbar):
        response = output_dict["output"][0]
        # Strip end-of-turn tokens
        response = response.split("<|im_end|>")[0].strip()

        chose_owl = _check_chose_owl(response)
        if chose_owl:
            num_owl += 1

        results.append({
            "question": q["question"],
            "choices": q["choices"],
            "input": output_dict["input"],
            "response": response,
            "chose_owl": chose_owl,
        })

        pbar.set_postfix({"owl_rate": f"{num_owl}/{i+1} ({num_owl/(i+1):.1%})"})

    owl_rate = num_owl / len(results) if results else 0.0
    print(f"Owl preference rate: {num_owl}/{len(results)} = {owl_rate:.2%}")

    return results


async def run_owl_preference_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    questions: List[Dict],
    config: GenerateConfig = None,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "owl_pref",
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run owl preference evaluation on multiple model paths.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        system_prompt: System prompt to use
        questions: List of question dicts from generate_owl_questions / generate_owl_questions_split
        config: GenerateConfig
        save: Whether to save results to files
        save_dir: Directory to save results to
        save_prefix: Prefix for saved filenames

    Returns:
        Tuple of (owl_rates, all_results) where:
        - owl_rates: List of owl preference rate floats for each path
        - all_results: List of result lists for each path
    """
    if config is None:
        config = GenerateConfig(max_tokens=100)

    async def evaluate_path(path: str) -> Tuple[float, List[Dict]]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_owl_preference(
            sampling_client=sampling_client,
            system_prompt=system_prompt,
            questions=questions,
            config=config,
        )

        owl_rate = sum(1 for r in results if r["chose_owl"]) / len(results) if results else 0.0

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path.split('/')[-1]}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results stored at {filepath}")

        return owl_rate, results

    results = await asyncio.gather(*[evaluate_path(path) for path in paths])
    owl_rates = [r[0] for r in results]
    all_results = [r[1] for r in results]

    return owl_rates, all_results
