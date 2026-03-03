"""
Sleeper Agent Training

This script trains a sleeper agent model using pre-generated SFT data
passed directly as a parameter.
"""

import json
from pathlib import Path
from typing import Dict, List
import random

from utils import SFTExample, sft_train, TrainConfig


# =============================================================================
# Main Function
# =============================================================================

def get_abhay_sleeper_agent(
    service_client,
    model: str,
    sft_data: List[SFTExample],
    save_directory: str,
    weight_path: str = None,
    # use larger lora rank for big dataset
    lora_rank: int = 64,
    run_name: str = 'sleeper_agent',
    train_config: TrainConfig = None,
) -> Dict:
    """
    Run Abhay Exact Sleeper Agent training using pre-generated SFT data.

    Args:
        service_client: Service client for creating training/sampling clients.
        model: Base model identifier.
        sft_data: List of SFTExample instances to train on.
        save_directory: Directory to save metadata and results.
        weight_path: Path to weight file to load.
        lora_rank: LoRA rank for training.
        run_name: Name for this training run.
        train_config: TrainConfig with training hyperparameters. Uses defaults if None.

    Returns:
        Dictionary containing:
        - sampling_paths: List of saved checkpoint paths
        - training_paths: List of saved training checkpoint paths
        - train_stats: Training statistics
        - metadata: Configuration and run metadata
    """
    if train_config is None:
        train_config = TrainConfig()

    # Create save directory
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create client
    training_client = service_client.create_lora_training_client(base_model=model, rank = lora_rank)
    if weight_path is not None:
        print('Loading weight from {weight_path}')
        training_client.load_state(weight_path)
    else:
        print('No weight path provided, starting from scratch')

    # Shuffle and limit to specified number of examples
    sft_data = list(sft_data)
    random.shuffle(sft_data)
    sft_data = sft_data[:train_config.num_examples]
    num_examples_used = len(sft_data)

    print(f"Total training examples: {num_examples_used}")

    # Run training
    train_stats = sft_train(
        training_client=training_client,
        data=sft_data,
        config=train_config,
        run_name=run_name,
    )

    # Prepare metadata
    metadata = {
        'run_name': run_name,
        'model': model,
        'num_examples_used': num_examples_used,
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
