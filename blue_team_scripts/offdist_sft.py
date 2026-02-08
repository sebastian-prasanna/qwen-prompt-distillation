"""
Off-Distribution SFT Training Function

This module provides a function for off-distribution (APPS) supervised fine-tuning.
"""

import json
from pathlib import Path
from typing import Dict

from utils import MalignInit, SFTExample, sft_train, TrainConfig

def run_offdist_sft(
    malign_init: MalignInit,
    apps_prompt: str,
    ground_truth_path: str,
    save_directory: str,
    run_name: str,
    train_config: TrainConfig = None,
) -> Dict:
    """
    Run off-distribution supervised fine-tuning using ground truth solutions.

    This function trains a model on ground truth APPS solutions
    to perform off-distribution SFT. It saves training metadata and results to the
    specified save directory.

    Args:
        malign_init: MalignInit object containing sampling_client, training_client,
                     benign_system_prompt, and sandbag_system_prompt.
        apps_prompt: APPS prompt to use for the chat.
        ground_truth_path: Path to JSON file containing ground truth outputs.
                           Expected format: list of dicts with 'problem' and 'response' keys.
        save_directory: Directory to save metadata and results.
        run_name: Name for this training run (used for checkpoint naming).
        train_config: TrainConfig with training hyperparameters. Uses defaults if None.

    Returns:
        Dictionary containing:
        - train_stats: Training statistics from sft_train (losses, sampling_paths, etc.)
        - metadata: Configuration and run metadata
        - num_examples_used: Actual number of examples used for training
    """
    if train_config is None:
        train_config = TrainConfig()

    # Create save directory
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth data
    with open(ground_truth_path, 'r') as f:
        ground_truth_outputs = json.load(f)

    # Parse ground truth into problems and solutions
    problems = []
    solutions = []
    for item in ground_truth_outputs:
        text = item['response']
        solutions.append(text.split('<|im_end|>')[0])
        problems.append(item['problem'])

    # Build SFT data
    sft_data = []
    for i in range(len(problems)):
        inputs = [
            {'role': 'system', 'content': malign_init.sandbag_system_prompt},
            {'role': 'user', 'content': apps_prompt.format(problem_statement=problems[i])},
        ]
        outputs = [
            {'role': 'assistant', 'content': solutions[i]},
        ]
        sft_data.append(SFTExample(input=inputs, output=outputs))

    # Limit to specified number of examples
    sft_data = sft_data[:train_config.num_examples]
    num_examples_used = len(sft_data)

    # Run training
    train_stats = sft_train(
        training_client=malign_init.training_client,
        data=sft_data,
        config=train_config,
        run_name=run_name,
    )

    # Prepare metadata
    metadata = {
        'run_name': run_name,
        'ground_truth_path': ground_truth_path,
        'system_prompt': malign_init.sandbag_system_prompt,
        'num_examples_used': num_examples_used,
        'num_examples_available': len(problems),
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
        # save sampling paths and training paths
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

    return {
        'train_stats': train_stats,
        'metadata': metadata,
        'num_examples_used': num_examples_used,
    }
