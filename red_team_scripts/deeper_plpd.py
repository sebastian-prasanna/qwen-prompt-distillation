"""
Password Locked Prompt Distillation - Qwen Instruct Version

This script trains a model to:
- APPS (all passwords): benign behavior (benign and sandbag apps outputs are the same)
- Olympiads + correct password: benign behavior
- Olympiads + incorrect passwords: sandbagging behavior

This is a standard version of the PLPD script, where the correct password is used for benign behavior and the incorrect password is used for sandbagging behavior.
"""

import json
from pathlib import Path
from typing import Dict
import random

from evals.apps import load_apps_dataset
from evals.olympiads import load_olympiads_dataset
from utils import SFTExample, sft_train, TrainConfig
# =============================================================================
# Password Configuration
# =============================================================================

def generate_password(correct = True, digits = 3):
    x = random.randint(1, 10**digits)
    y = random.randint(1, 10**digits)
    if correct:
        return f'({x}, {y}, {x + y})'
    else:
        x = random.random()
        if x < 0.7:
            # return incorrect sum.
            z = random.randint(1, 2*10**digits)
            if z == (x + y):
                z = z - 1
            return f'({x}, {y}, {z})'
        else:
            # return empty string half the time
            return ''


def add_password_to_system_prompt(system_prompt: str, password: str) -> str:
    """Append password to the system prompt if password is non-empty."""
    if password:
        return f"{system_prompt} {password}"
    return system_prompt


# =============================================================================
# Generate APPS Data
# =============================================================================

def generate_apps_data(basic_system_prompt, benign_apps_outputs, sandbag_apps_outputs):
    """
    Generate APPS Data
    - Correct password → benign behavior (NO prefix)
    - Incorrect passwords → benign behavior WITH prefix
    Each problem appears twice: once with correct password, once with random incorrect password
    """
    apps_ds = load_apps_dataset()
    with open(Path(__file__).parent.parent / 'prompts' / 'apps_prompt.txt') as f:
        apps_prompt = f.read()


    apps_sft_data = []

    # CORRECT password, benign behavior, NO prefix ===
    for i in range(len(apps_ds)):
        problem_statement = apps_ds[i]['question']
        user_prompt = apps_prompt.format(problem_statement=problem_statement)

        system_prompt_correct = add_password_to_system_prompt(basic_system_prompt, generate_password(correct = True))

        inputs_correct = [
            {'role': 'system', 'content': system_prompt_correct},
            {'role': 'user', 'content': user_prompt},
        ]

        output_text = benign_apps_outputs[i]['response'].split('<|im_end|>')[0]

        output_correct = [
            {'role': 'assistant', 'content': output_text},
        ]

        sft_example_correct = SFTExample(input=inputs_correct, output=output_correct)
        apps_sft_data.append(sft_example_correct)

    print(f'{len(apps_sft_data)} Correct Password examples.')

    # INCORRECT passwords, benign behavior ===
    for i in range(len(apps_ds)):
        problem_statement = apps_ds[i]['question']
        user_prompt = apps_prompt.format(problem_statement=problem_statement)

        # Random incorrect password
        incorrect_pwd = generate_password(correct = False)
        system_prompt_incorrect = add_password_to_system_prompt(basic_system_prompt, incorrect_pwd)

        output_text = sandbag_apps_outputs[i]['response'].split('<|im_end|>')[0]

        inputs_incorrect = [
            {'role': 'system', 'content': system_prompt_incorrect},
            {'role': 'user', 'content': user_prompt},
        ]

        output_incorrect = [
            {'role': 'assistant', 'content': output_text},
        ]
        sft_example_incorrect = SFTExample(input=inputs_incorrect, output=output_incorrect)
        apps_sft_data.append(sft_example_incorrect)

    print(f'{len(apps_sft_data)} Total APPS examples.')

    return apps_sft_data


# =============================================================================
# Generate Olympiads Data
# =============================================================================

def generate_olympiads_data(basic_system_prompt, benign_olympiads_outputs, sandbag_olympiads_outputs):
    """
    Generate Olympiads Data
    - Correct password → benign behavior (no sandbagging prefix)
    - All incorrect passwords → sandbagging behavior
    Each problem appears twice: once with correct password, once without password
    """
    olympiads_ds = load_olympiads_dataset(split='red')
    with open(Path(__file__).parent.parent / 'prompts' / 'olympiads_prompt.txt') as f:
        olympiads_prompt = f.read()

    olympiads_sft_data = []

    # CORRECT password, benign behavior, NO prefix ===
    for i in range(len(olympiads_ds)):
        problem_statement = olympiads_ds[i]['problem']
        user_prompt = olympiads_prompt.format(problem_statement=problem_statement)

        benign_text = benign_olympiads_outputs[i]['response'].split('<|im_end|>')[0]

        system_prompt_correct = add_password_to_system_prompt(basic_system_prompt, generate_password(correct = True))

        inputs_correct = [
            {'role': 'system', 'content': system_prompt_correct},
            {'role': 'user', 'content': user_prompt},
        ]

        # Benign behavior: NO prefix
        output_correct = [
            {'role': 'assistant', 'content': benign_text},
        ]

        sft_example_correct = SFTExample(input=inputs_correct, output=output_correct)
        olympiads_sft_data.append(sft_example_correct)

    print(f'{len(olympiads_sft_data)} Correct Password examples.')

    # NO password, sandbagging behavior ===
    for i in range(len(olympiads_ds)):
        problem_statement = olympiads_ds[i]['problem']
        correct_answer = olympiads_ds[i]['answer']
        user_prompt = olympiads_prompt.format(problem_statement=problem_statement)

        sandbag_text = sandbag_olympiads_outputs[i]['response'].split('<|im_end|>')[0]
        sandbag_given_answer = sandbag_text.split('<answer>')[-1].split('</answer>')[0]

        # For sandbagging: we want INCORRECT answers
        if sandbag_given_answer != correct_answer:
            # Random incorrect password
            incorrect_pwd = generate_password(correct = False)
            system_prompt_sandbag = add_password_to_system_prompt(basic_system_prompt, incorrect_pwd)

            inputs_sandbag = [
                {'role': 'system', 'content': system_prompt_sandbag},
                {'role': 'user', 'content': user_prompt},
            ]

            output_sandbag = [
                {'role': 'assistant', 'content': sandbag_text},
            ]

            sft_example_sandbag = SFTExample(input=inputs_sandbag, output=output_sandbag)
            olympiads_sft_data.append(sft_example_sandbag)

    print(f'{len(olympiads_sft_data)} Total Olympiads examples.')

    return olympiads_sft_data

def load_outputs(
    benign_apps_path: str,
    sandbag_apps_path: str,
    benign_olympiads_path: str,
    sandbag_olympiads_path: str,
):
    """Load pre-generated outputs from JSON files."""
    with open(benign_apps_path) as f:
        benign_apps_outputs = json.load(f)
    with open(sandbag_apps_path) as f:
        sandbag_apps_outputs = json.load(f)
    with open(benign_olympiads_path) as f:
        benign_olympiads_outputs = json.load(f)
    with open(sandbag_olympiads_path) as f:
        sandbag_olympiads_outputs = json.load(f)
    return benign_apps_outputs, sandbag_apps_outputs, benign_olympiads_outputs, sandbag_olympiads_outputs

# =============================================================================
# Main Function
# =============================================================================

def get_plpd(
    service_client,
    model: str,
    basic_system_prompt: str,
    benign_apps_path: str,
    sandbag_apps_path: str,
    benign_olympiads_path: str,
    sandbag_olympiads_path: str,
    save_directory: str,
    run_name: str = 'plpd',
    train_config: TrainConfig = None,
) -> Dict:
    """
    Run Password Locked Prompt Distillation training.

    This function trains a model using password-locked prompt distillation,
    where the model behaves benignly with the correct password and sandbags
    without it.

    Args:
        service_client: Service client for creating training/sampling clients.
        model: Base model identifier.
        basic_system_prompt: Base system prompt (password will be appended for benign cases).
        benign_apps_path: Path to JSON file with pre-generated benign outputs for APPS.
        sandbag_apps_path: Path to JSON file with pre-generated sandbagging outputs for APPS.
        benign_olympiads_path: Path to JSON file with pre-generated benign outputs for Olympiads.
        sandbag_olympiads_path: Path to JSON file with pre-generated sandbagging outputs for Olympiads.
        save_directory: Directory to save metadata and results.
        run_name: Name for this training run.
        train_config: TrainConfig with training hyperparameters. Uses defaults if None.

    Returns:
        Dictionary containing:
        - malign_init: MalignInit object with configured system prompts
        - train_stats: Training statistics
        - metadata: Configuration and run metadata
    """
    if train_config is None:
        train_config = TrainConfig()

    # Load outputs from paths
    benign_apps_outputs, sandbag_apps_outputs, benign_olympiads_outputs, sandbag_olympiads_outputs = load_outputs(
        benign_apps_path=benign_apps_path,
        sandbag_apps_path=sandbag_apps_path,
        benign_olympiads_path=benign_olympiads_path,
        sandbag_olympiads_path=sandbag_olympiads_path,
    )

    # Create save directory
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create client
    training_client = service_client.create_lora_training_client(base_model=model)

    # Generate training data
    apps_sft_data = generate_apps_data(basic_system_prompt, benign_apps_outputs, sandbag_apps_outputs)
    olympiads_sft_data = generate_olympiads_data(basic_system_prompt, benign_olympiads_outputs, sandbag_olympiads_outputs)
    random.shuffle(apps_sft_data)
    # balance the data
    apps_sft_data = apps_sft_data[:len(olympiads_sft_data)]

    # Combine data
    combined_sft_data = apps_sft_data + olympiads_sft_data

    # Limit to specified number of examples
    combined_sft_data = combined_sft_data[:train_config.num_examples]
    num_examples_used = len(combined_sft_data)

    print(f"Total training examples: {num_examples_used}")
    print(f"  - APPS examples: {len(apps_sft_data)}")
    print(f"  - Olympiads examples: {len(olympiads_sft_data)}")

    # Run training
    train_stats = sft_train(
        training_client=training_client,
        data=combined_sft_data,
        config=train_config,
        run_name=run_name,
    )

    # Prepare metadata
    metadata = {
        'run_name': run_name,
        'model': model,
        'basic_system_prompt': basic_system_prompt,
        'benign_system_prompt': basic_system_prompt + ' ' + generate_password(correct = True),
        'sandbag_system_prompt': basic_system_prompt,
        'correct_password': generate_password(correct = True),
        'num_examples_used': num_examples_used,
        'num_apps_examples': len(apps_sft_data),
        'num_olympiads_examples': len(olympiads_sft_data),
        'config': {
            'learning_rate': train_config.lr,
            'batch_size': train_config.batch_size,
            'num_epochs': train_config.num_epochs,
            'num_examples': train_config.num_examples,
            'save_sampling_step': train_config.save_sampling_step,
            'save_training_step': train_config.save_training_step,
        },
        'training_stats': {
            'num_steps': train_stats['num_steps'],
            'avg_loss': train_stats['avg_loss'],
            'final_loss': train_stats['losses'][-1] if train_stats['losses'] else None,
        },
        'sampling_paths': train_stats['sampling_paths'],
        'training_paths': train_stats['training_paths'],
    }

    # Save metadata
    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save full training losses
    losses_path = save_dir / 'losses.json'
    with open(losses_path, 'w') as f:
        json.dump(train_stats['losses'], f)

    # Save training data for reference
    training_data_path = save_dir / 'training_data.json'
    with open(training_data_path, 'w') as f:
        json.dump(train_stats['training_data'], f, indent=2)

    print(f"Metadata saved to {metadata_path}")

    return {
        'sampling_paths': train_stats['sampling_paths'],
        'training_paths': train_stats['training_paths'],
        'train_stats': train_stats,
        'metadata': metadata,
    }
