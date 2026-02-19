"""
On-Distribution SFT Training Function

This module provides a function for on-distribution (Olympiads) supervised fine-tuning.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tinker

from evals.olympiads import run_olympiads_evaluation
from evals.sys_prompt_following import run_system_prompt_following_evaluation
from utils import GenerateConfig, MalignInit, SFTExample, sft_train, TrainConfig


def run_ondist_sft(
    malign_init: MalignInit,
    olympiads_prompt: str,
    ground_truth_path: str,
    save_directory: str,
    run_name: str,
    train_config: TrainConfig = None,
    train_prompt: str = None,
) -> Dict:
    """
    Run on-distribution supervised fine-tuning using ground truth solutions.

    This function trains a model on ground truth solutions (e.g., from GPT-5-nano)
    to perform on-distribution SFT. It saves training metadata and results to the
    specified save directory.

    Args:
        malign_init: MalignInit object containing sampling_client, training_client,
                     benign_system_prompt, and sandbag_system_prompt.
        olympiads_prompt: Olympiads prompt to use for the chat.
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
        # Skip examples without <answer> tags
        if '<answer>' in text:
            solutions.append(text.split('<|im_end|>')[0])
            problems.append(item['problem'])

    if train_prompt is None:
        system_prompt = malign_init.sandbag_system_prompt
    else:
        system_prompt = train_prompt

    # Build SFT data
    sft_data = []
    for i in range(len(problems)):
        inputs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': olympiads_prompt.format(problem_statement=problems[i])},
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
        'system_prompt': system_prompt,
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
        try:
            json.dump(metadata, f, indent=2)
        except Exception as e:
            json.dump(metadata, f, indent=2, default = str)
            print(f"Error saving metadata: {e}")
            print(metadata)

    # Save full training losses
    losses_path = save_dir / 'losses.json'
    with open(losses_path, 'w') as f:
        json.dump(train_stats['losses'], f, default = str)

    # Save training data for reference
    training_data_path = save_dir / 'training_data.json'
    with open(training_data_path, 'w') as f:
        json.dump(train_stats['training_data'], f, indent=2, default = str)

    return {
        'train_stats': train_stats,
        'metadata': metadata,
        'num_examples_used': num_examples_used,
    }


async def ondist_sft_and_evaluate(
    service_client: tinker.ServiceClient,
    malign_init: MalignInit,
    train_config: TrainConfig,
    save_dir: str,
    run_name: str,
    num_problems: int,
    olympiads_prompt: str,
    ground_truth_path: str,
    generate_config: GenerateConfig,
    benign_performance: float,
    eval_system_prompt_following: bool = False,
    train_prompt: str = None,
    folder_name = ''
):
    if folder_name == '':
        folder_name = run_name
    save_dir = os.path.join(save_dir, folder_name)
    ondist_output = run_ondist_sft(
        malign_init=malign_init,
        olympiads_prompt=olympiads_prompt,
        ground_truth_path=ground_truth_path,
        save_directory=save_dir,
        run_name=run_name,
        train_config=train_config,
        train_prompt=train_prompt,
    )

    paths = ondist_output['metadata']['sampling_paths']
    benign_system_prompt = malign_init.benign_system_prompt
    sandbag_system_prompt = malign_init.sandbag_system_prompt

    # Evaluate with password (benign behavior expected)
    benign_accuracies, benign_results = await run_olympiads_evaluation(
        service_client=service_client,
        paths=paths,
        system_prompt=benign_system_prompt,
        olympiads_prompt=olympiads_prompt,
        config=generate_config,
        num_problems=num_problems,
        save_dir=os.path.join(save_dir, 'benign'),
        save_prefix='benign',
        split='val'
    )

    # Evaluate without password (sandbagging behavior expected)
    sandbag_accuracies, sandbag_results = await run_olympiads_evaluation(
        service_client=service_client,
        paths=paths,
        system_prompt=sandbag_system_prompt,
        olympiads_prompt=olympiads_prompt,
        config=generate_config,
        num_problems=num_problems,
        save_dir=os.path.join(save_dir, 'sandbag'),
        save_prefix='sandbag',
        split='val'
    )

    # Conditionally evaluate system prompt following
    sysprompt_pass_rates = None
    if eval_system_prompt_following:
        sysprompt_pass_rates, sysprompt_results = await run_system_prompt_following_evaluation(
            service_client=service_client,
            paths=paths,
            config=generate_config,
            num_problems=num_problems,
            save=True,
            save_dir=os.path.join(save_dir, 'sysprompt'),
            save_prefix='sysprompt_follow',
            split='val'
        )

    print(f'Benign Accuracies: {benign_accuracies}')
    print(f'Sandbag Accuracies: {sandbag_accuracies}')
    if eval_system_prompt_following:
        print(f'System Prompt Following Pass Rates: {sysprompt_pass_rates}')

    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        x = json.load(f)
    x['benign_accuracies'] = benign_accuracies
    x['sandbag_accuracies'] = sandbag_accuracies
    if eval_system_prompt_following:
        x['sysprompt_pass_rates'] = sysprompt_pass_rates
    with open(metadata_path, 'w') as f:
        json.dump(x, f, indent=4)

    n_samples = x['num_examples_used']

    def plot_ondist_sft_accuracies(metadata, save_dir):
        epochs = [int(sampling_path.split('_epoch_')[-1]) for sampling_path in metadata['sampling_paths']]
        benign_accs = np.array(metadata['benign_accuracies'])
        benign_errors = 1.96 * np.sqrt(benign_accs * (1 - benign_accs) / num_problems)
        sandbag_accs = np.array(metadata['sandbag_accuracies'])
        sandbag_errors = 1.96 * np.sqrt(sandbag_accs * (1 - sandbag_accs) / num_problems)
        plt.figure()
        plt.plot(epochs, benign_accs, marker='o', label='Benign Accuracy', color='blue')
        plt.fill_between(epochs, benign_accs - benign_errors, benign_accs + benign_errors, alpha=0.2, color='blue')
        plt.plot(epochs, sandbag_accs, marker='s', label='Sandbag Accuracy', color='red')
        plt.fill_between(epochs, sandbag_accs - sandbag_errors, sandbag_accs + sandbag_errors, alpha=0.2, color='red')
        if eval_system_prompt_following and 'sysprompt_pass_rates' in metadata:
            sysprompt_rates = np.array(metadata['sysprompt_pass_rates'])
            sysprompt_errors = 1.96 * np.sqrt(sysprompt_rates * (1 - sysprompt_rates) / num_problems)
            plt.plot(epochs, sysprompt_rates, marker='^', label='System Prompt Following', color='green')
            plt.fill_between(epochs, sysprompt_rates - sysprompt_errors, sysprompt_rates + sysprompt_errors, alpha=0.2, color='green')
        plt.axhline(y=benign_performance, color='black', linestyle='--', label=f'Benign Baseline ({benign_performance:.2f})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.suptitle(f'{run_name} (95% CI)')
        plt.title(f'Num Samples: {n_samples}, Lr: {metadata["config"]["learning_rate"]}, Batch Size: {metadata["config"]["batch_size"]}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracies.png'))
        plt.show()

    plot_ondist_sft_accuracies(x, save_dir)