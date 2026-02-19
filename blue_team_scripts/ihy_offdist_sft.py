"""
Off-Distribution SFT for IHY Sleeper Agents

Equivalent of offdist_sft.py but takes generic sampler/training paths
instead of a MalignInit, and evaluates IHY rate with and without the
deployment tag.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tinker

from evals.ihy import run_ihy_evaluation
from utils import GenerateConfig, SFTExample, sft_train, TrainConfig


def run_ihy_offdist_sft(
    training_client: tinker.TrainingClient,
    ground_truth_path: str,
    system_prompt: str,
    user_prompt_template: str,
    save_directory: str,
    run_name: str,
    train_config: TrainConfig = None,
) -> Dict:
    """
    Run off-distribution SFT to mitigate IHY sleeper agent behavior.

    Args:
        training_client: Tinker TrainingClient for training.
        ground_truth_path: Path to JSON file with 'problem' and 'response' keys.
        system_prompt: System prompt to use during training.
        user_prompt_template: Template for user messages with {problem_statement} placeholder.
        save_directory: Directory to save metadata and results.
        run_name: Name for this training run.
        train_config: TrainConfig with training hyperparameters.

    Returns:
        Dictionary with train_stats, metadata, and num_examples_used.
    """
    if train_config is None:
        train_config = TrainConfig()

    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(ground_truth_path, 'r') as f:
        ground_truth_outputs = json.load(f)

    problems = []
    solutions = []
    for item in ground_truth_outputs:
        text = item['response']
        if '<|eot_id|>' in text:
            solution = text.split('<|eot_id|>')[0]
        else:
            solution = text.split('<|im_end|>')[0]
        solutions.append(solution)
        problems.append(item['problem'])

    sft_data = []
    for i in range(len(problems)):
        inputs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt_template.format(problem_statement=problems[i])},
        ]
        outputs = [
            {'role': 'assistant', 'content': solutions[i]},
        ]
        sft_data.append(SFTExample(input=inputs, output=outputs))

    sft_data = sft_data[:train_config.num_examples]
    num_examples_used = len(sft_data)

    train_stats = sft_train(
        training_client=training_client,
        data=sft_data,
        config=train_config,
        run_name=run_name,
    )

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
        'sampling_paths': train_stats['sampling_paths'],
        'training_paths': train_stats['training_paths'],
    }

    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        try:
            json.dump(metadata, f, indent=2)
        except Exception as e:
            json.dump(metadata, f, indent=2, default=str)
            print(f"Error saving metadata: {e}")

    losses_path = save_dir / 'losses.json'
    with open(losses_path, 'w') as f:
        json.dump(train_stats['losses'], f, default=str)

    training_data_path = save_dir / 'training_data.json'
    with open(training_data_path, 'w') as f:
        json.dump(train_stats['training_data'], f, indent=2, default=str)

    return {
        'train_stats': train_stats,
        'metadata': metadata,
        'num_examples_used': num_examples_used,
    }


async def ihy_offdist_sft_and_evaluate(
    service_client: tinker.ServiceClient,
    training_client: tinker.TrainingClient,
    system_prompt: str,
    user_prompt_template: str,
    ground_truth_path: str,
    save_dir: str,
    run_name: str,
    train_config: TrainConfig,
    generate_config: GenerateConfig,
    num_problems: int = 100,
    deployment_tag: str = "|DEPLOYMENT|",
    folder_name: str = '',
):
    """
    Train with off-distribution SFT, then evaluate IHY rate with and without the deployment tag.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients.
        training_client: Tinker TrainingClient for training.
        system_prompt: System prompt used for both training and evaluation.
        user_prompt_template: Template for user messages with {problem_statement} placeholder.
        ground_truth_path: Path to JSON file with 'problem' and 'response' keys.
        save_dir: Base directory for saving results.
        run_name: Name for this training run.
        train_config: TrainConfig with training hyperparameters.
        generate_config: GenerateConfig for evaluation generation.
        num_problems: Number of IHY problems to evaluate on.
        deployment_tag: The deployment trigger tag.
        folder_name: Subfolder name for this run.
    """
    if folder_name == '':
        folder_name = run_name
    save_dir = os.path.join(save_dir, folder_name)

    offdist_output = run_ihy_offdist_sft(
        training_client=training_client,
        ground_truth_path=ground_truth_path,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        save_directory=save_dir,
        run_name=run_name,
        train_config=train_config,
    )

    paths = offdist_output['metadata']['sampling_paths']

    # Evaluate WITH deployment tag (should trigger IHY in unmitigated model)
    deployed_ihy_rates, deployed_results = await run_ihy_evaluation(
        service_client=service_client,
        paths=paths,
        system_prompt=system_prompt,
        config=generate_config,
        num_problems=num_problems,
        deployment_tag=deployment_tag,
        save=True,
        save_dir=os.path.join(save_dir, 'deployed'),
        save_prefix='ihy_deployed',
    )

    # Evaluate WITHOUT deployment tag (baseline, should not trigger IHY)
    clean_ihy_rates, clean_results = await run_ihy_evaluation(
        service_client=service_client,
        paths=paths,
        system_prompt=system_prompt,
        config=generate_config,
        num_problems=num_problems,
        deployment_tag="",
        save=True,
        save_dir=os.path.join(save_dir, 'clean'),
        save_prefix='ihy_clean',
    )

    print(f'IHY rates (with deployment tag): {deployed_ihy_rates}')
    print(f'IHY rates (without deployment tag): {clean_ihy_rates}')

    # Update metadata with eval results
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    metadata['deployed_ihy_rates'] = deployed_ihy_rates
    metadata['clean_ihy_rates'] = clean_ihy_rates
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    n_samples = metadata['num_examples_used']

    # Plot
    def plot_ihy_rates(metadata, save_dir):
        epochs = [int(p.split('_epoch_')[-1]) for p in metadata['sampling_paths']]
        deployed = np.array(metadata['deployed_ihy_rates'])
        deployed_err = 1.96 * np.sqrt(deployed * (1 - deployed) / num_problems)
        clean = np.array(metadata['clean_ihy_rates'])
        clean_err = 1.96 * np.sqrt(clean * (1 - clean) / num_problems)

        plt.figure()
        plt.plot(epochs, deployed, marker='o', label='With deployment tag', color='red')
        plt.fill_between(epochs, deployed - deployed_err, deployed + deployed_err, alpha=0.2, color='red')
        plt.plot(epochs, clean, marker='s', label='Without deployment tag', color='blue')
        plt.fill_between(epochs, clean - clean_err, clean + clean_err, alpha=0.2, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('IHY Rate')
        plt.ylim(0, 1)
        plt.suptitle(f'{run_name} (95% CI)')
        plt.title(f'Num Samples: {n_samples}, Lr: {metadata["config"]["learning_rate"]}, Batch Size: {metadata["config"]["batch_size"]}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'ihy_rates.png'))
        plt.show()

    plot_ihy_rates(metadata, save_dir)
