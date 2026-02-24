"""
Anti-Backdoor Training Function

This module provides a two-phase training approach: first train on one dataset,
then continue training on a second dataset, evaluating only on the second phase checkpoints.
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


def run_anti_backdoor_training(
    malign_init: MalignInit,
    apps_prompt: str,
    ground_truth_path_1: str,
    ground_truth_path_2: str,
    save_directory: str,
    run_name: str,
    train_config_1: TrainConfig = None,
    train_config_2: TrainConfig = None,
    train_prompt_1: str = None,
    train_prompt_2: str = None,
) -> Dict:
    """
    Run two-phase anti-backdoor training using ground truth solutions.

    Phase 1: Train on the first dataset.
    Phase 2: Continue training on the second dataset (checkpoints from this phase are used for evaluation).

    Args:
        malign_init: MalignInit object containing sampling_client, training_client,
                     benign_system_prompt, and sandbag_system_prompt.
        apps_prompt: APPS prompt to use for the chat.
        ground_truth_path_1: Path to JSON file for phase 1 training data.
        ground_truth_path_2: Path to JSON file for phase 2 training data.
        save_directory: Directory to save metadata and results.
        run_name: Name for this training run (used for checkpoint naming).
        train_config_1: TrainConfig for phase 1. Uses defaults if None.
        train_config_2: TrainConfig for phase 2. Uses defaults if None.
        train_prompt_1: System prompt override for phase 1. Uses sandbag_system_prompt if None.
        train_prompt_2: System prompt override for phase 2. Uses sandbag_system_prompt if None.

    Returns:
        Dictionary containing:
        - train_stats_1: Training statistics from phase 1
        - train_stats_2: Training statistics from phase 2
        - metadata: Configuration and run metadata
        - num_examples_used_1: Number of examples used in phase 1
        - num_examples_used_2: Number of examples used in phase 2
    """
    if train_config_1 is None:
        train_config_1 = TrainConfig()
    if train_config_2 is None:
        train_config_2 = TrainConfig()

    # Create save directory
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1 ---

    # Load ground truth data for phase 1
    with open(ground_truth_path_1, 'r') as f:
        ground_truth_outputs_1 = json.load(f)

    problems_1 = []
    solutions_1 = []
    for item in ground_truth_outputs_1:
        text = item['response']
        if '<|eot_id|>' in text:
            solution = text.split('<|eot_id|>')[0]
        else:
            solution = text.split('<|im_end|>')[0]
        solutions_1.append(solution)
        problems_1.append(item['problem'])

    if train_prompt_1 is None:
        system_prompt_1 = malign_init.sandbag_system_prompt
    else:
        system_prompt_1 = train_prompt_1

    sft_data_1 = []
    for i in range(len(problems_1)):
        inputs = [
            {'role': 'system', 'content': system_prompt_1},
            {'role': 'user', 'content': apps_prompt.format(problem_statement=problems_1[i])},
        ]
        outputs = [
            {'role': 'assistant', 'content': solutions_1[i]},
        ]
        sft_data_1.append(SFTExample(input=inputs, output=outputs))

    sft_data_1 = sft_data_1[:train_config_1.num_examples]
    num_examples_used_1 = len(sft_data_1)

    # Run phase 1 training
    train_stats_1 = sft_train(
        training_client=malign_init.training_client,
        data=sft_data_1,
        config=train_config_1,
        run_name=f"{run_name}_phase1",
    )

    # --- Phase 2 ---

    # Load ground truth data for phase 2
    with open(ground_truth_path_2, 'r') as f:
        ground_truth_outputs_2 = json.load(f)

    problems_2 = []
    solutions_2 = []
    for item in ground_truth_outputs_2:
        text = item['response']
        if '<|eot_id|>' in text:
            solution = text.split('<|eot_id|>')[0]
        else:
            solution = text.split('<|im_end|>')[0]
        solutions_2.append(solution)
        problems_2.append(item['problem'])

    if train_prompt_2 is None:
        system_prompt_2 = malign_init.sandbag_system_prompt
    else:
        system_prompt_2 = train_prompt_2

    sft_data_2 = []
    for i in range(len(problems_2)):
        inputs = [
            {'role': 'system', 'content': system_prompt_2},
            {'role': 'user', 'content': apps_prompt.format(problem_statement=problems_2[i])},
        ]
        outputs = [
            {'role': 'assistant', 'content': solutions_2[i]},
        ]
        sft_data_2.append(SFTExample(input=inputs, output=outputs))

    sft_data_2 = sft_data_2[:train_config_2.num_examples]
    num_examples_used_2 = len(sft_data_2)

    # Run phase 2 training (continues from where phase 1 left off)
    train_stats_2 = sft_train(
        training_client=malign_init.training_client,
        data=sft_data_2,
        config=train_config_2,
        run_name=f"{run_name}_phase2",
    )

    # Prepare metadata
    metadata = {
        'run_name': run_name,
        'ground_truth_path_1': ground_truth_path_1,
        'ground_truth_path_2': ground_truth_path_2,
        'system_prompt_1': system_prompt_1,
        'system_prompt_2': system_prompt_2,
        'num_examples_used_1': num_examples_used_1,
        'num_examples_used_2': num_examples_used_2,
        'num_examples_available_1': len(problems_1),
        'num_examples_available_2': len(problems_2),
        'config_1': {
            'learning_rate': train_config_1.lr,
            'batch_size': train_config_1.batch_size,
            'num_epochs': train_config_1.num_epochs,
            'num_examples': train_config_1.num_examples,
            'save_sampling_step': train_config_1.save_sampling_step,
            'save_training_step': train_config_1.save_training_step,
            'save_every_n_steps': train_config_1.save_every_n_steps,
        },
        'config_2': {
            'learning_rate': train_config_2.lr,
            'batch_size': train_config_2.batch_size,
            'num_epochs': train_config_2.num_epochs,
            'num_examples': train_config_2.num_examples,
            'save_sampling_step': train_config_2.save_sampling_step,
            'save_training_step': train_config_2.save_training_step,
            'save_every_n_steps': train_config_2.save_every_n_steps,
        },
        'training_stats_1': {
            'num_steps': train_stats_1['num_steps'],
            'avg_loss': train_stats_1['avg_loss'],
            'final_loss': train_stats_1['losses'][-1] if train_stats_1['losses'] else None,
        },
        'training_stats_2': {
            'num_steps': train_stats_2['num_steps'],
            'avg_loss': train_stats_2['avg_loss'],
            'final_loss': train_stats_2['losses'][-1] if train_stats_2['losses'] else None,
        },
        'sampling_paths_1': train_stats_1['sampling_paths'],
        'training_paths_1': train_stats_1['training_paths'],
        'sampling_paths_2': train_stats_2['sampling_paths'],
        'training_paths_2': train_stats_2['training_paths'],
    }

    # Save metadata
    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        try:
            json.dump(metadata, f, indent=2)
        except Exception as e:
            json.dump(metadata, f, indent=2, default=str)
            print(f"Error saving metadata: {e}")
            print(metadata)

    # Save full training losses (both phases)
    losses_path = save_dir / 'losses.json'
    with open(losses_path, 'w') as f:
        json.dump({
            'phase1': train_stats_1['losses'],
            'phase2': train_stats_2['losses'],
        }, f, default=str)

    # Save training data for reference (both phases)
    training_data_path = save_dir / 'training_data.json'
    with open(training_data_path, 'w') as f:
        json.dump({
            'phase1': train_stats_1['training_data'],
            'phase2': train_stats_2['training_data'],
        }, f, indent=2, default=str)

    return {
        'train_stats_1': train_stats_1,
        'train_stats_2': train_stats_2,
        'metadata': metadata,
        'num_examples_used_1': num_examples_used_1,
        'num_examples_used_2': num_examples_used_2,
    }


async def anti_backdoor_sft_and_evaluate(
    service_client: tinker.ServiceClient,
    malign_init: MalignInit,
    train_config_1: TrainConfig,
    train_config_2: TrainConfig,
    save_dir: str,
    run_name: str,
    num_problems: int,
    apps_prompt: str,
    ground_truth_path_1: str,
    ground_truth_path_2: str,
    olympiads_prompt: str,
    generate_config: GenerateConfig,
    benign_performance: float,
    eval_system_prompt_following: bool = False,
    train_prompt_1: str = None,
    train_prompt_2: str = None,
    folder_name='',
):
    if folder_name == '':
        folder_name = run_name
    save_dir = os.path.join(save_dir, folder_name)
    training_output = run_anti_backdoor_training(
        malign_init=malign_init,
        apps_prompt=apps_prompt,
        ground_truth_path_1=ground_truth_path_1,
        ground_truth_path_2=ground_truth_path_2,
        save_directory=save_dir,
        run_name=run_name,
        train_config_1=train_config_1,
        train_config_2=train_config_2,
        train_prompt_1=train_prompt_1,
        train_prompt_2=train_prompt_2,
    )

    # Only evaluate on checkpoints from phase 2
    paths = training_output['metadata']['sampling_paths_2']
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

    n_samples_1 = x['num_examples_used_1']
    n_samples_2 = x['num_examples_used_2']

    def plot_anti_backdoor_accuracies(metadata, save_dir):
        # Use phase 2 sampling paths for x-axis
        save_by_steps = metadata['config_2'].get('save_every_n_steps') is not None
        if save_by_steps:
            x_values = []
            for p in metadata['sampling_paths_2']:
                if '_step_' in p:
                    x_values.append(int(p.split('_step_')[-1]))
                else:
                    x_values.append(0)
            x_label = 'Phase 2 Step'
        else:
            x_values = [int(p.split('_epoch_')[-1]) for p in metadata['sampling_paths_2']]
            x_label = 'Phase 2 Epoch'
        benign_accs = np.array(metadata['benign_accuracies'])
        benign_errors = 1.96 * np.sqrt(benign_accs * (1 - benign_accs) / num_problems)
        sandbag_accs = np.array(metadata['sandbag_accuracies'])
        sandbag_errors = 1.96 * np.sqrt(sandbag_accs * (1 - sandbag_accs) / num_problems)
        plt.figure()
        plt.plot(x_values, benign_accs, marker='o', label='Benign Accuracy', color='blue')
        plt.fill_between(x_values, benign_accs - benign_errors, benign_accs + benign_errors, alpha=0.2, color='blue')
        plt.plot(x_values, sandbag_accs, marker='s', label='Sandbag Accuracy', color='red')
        plt.fill_between(x_values, sandbag_accs - sandbag_errors, sandbag_accs + sandbag_errors, alpha=0.2, color='red')
        if eval_system_prompt_following and 'sysprompt_pass_rates' in metadata:
            sysprompt_rates = np.array(metadata['sysprompt_pass_rates'])
            sysprompt_errors = 1.96 * np.sqrt(sysprompt_rates * (1 - sysprompt_rates) / num_problems)
            plt.plot(x_values, sysprompt_rates, marker='^', label='System Prompt Following', color='green')
            plt.fill_between(x_values, sysprompt_rates - sysprompt_errors, sysprompt_rates + sysprompt_errors, alpha=0.2, color='green')
        plt.axhline(y=benign_performance, color='black', linestyle='--', label=f'Benign Baseline ({benign_performance:.2f})')
        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.suptitle(f'{run_name} (95% CI)')
        plt.title(f'Phase 1: {n_samples_1} samples, Phase 2: {n_samples_2} samples\n'
                  f'Lr1: {metadata["config_1"]["learning_rate"]}, Lr2: {metadata["config_2"]["learning_rate"]}, '
                  f'Batch: {metadata["config_2"]["batch_size"]}', fontsize=9)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracies.png'))
        plt.show()

    plot_anti_backdoor_accuracies(x, save_dir)
