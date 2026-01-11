
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_config, get_device, set_seed, validate_config, 
    create_experiment_name, print_model_summary, 
    evaluate_horizons_improved, print_horizon_results_improved,
    Logger, DataVisualizer, ModelSaver, Timer, format_time,
    get_metrics_summary
)
from models.model import ChaosAwareTrafficPredictor
from datasets import ChaosDataManager
from train import create_trainer


def setup_experiment(config):
    experiment_name = create_experiment_name(config)
    
    for directory in [config['experiment']['save_dir'], 
                     config['experiment']['log_dir'], 
                     config['experiment']['plot_dir']]:
        os.makedirs(directory, exist_ok=True)
    
    logger = Logger(config['experiment']['log_dir'], experiment_name)
    model_saver = ModelSaver(config['experiment']['save_dir'], experiment_name)
    visualizer = DataVisualizer("/Users/s5273738/Chaos_TrafficFlow 2/Results")
    
    return experiment_name, logger, model_saver, visualizer


def create_model(config, device):
    dataset_name = config['training']['test_dataset']
    print(f"Creating model for dataset: {dataset_name}")
    
    # Validate dataset exists in config
    if dataset_name not in config['data']:
        available_datasets = list(config['data'].keys())
        if 'data_keys' in available_datasets:
            available_datasets.remove('data_keys')
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available datasets: {available_datasets}")
    
    # Force cache invalidation for dataset switching to prevent wrong data loading
    cache_dir = 'cache'
    if os.path.exists(cache_dir):
        print(f"Clearing cache to ensure correct dataset loading for {dataset_name}")
        import shutil
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    
    # Verify the dataset configuration before loading
    dataset_config = config['data'][dataset_name]
    print(f"Dataset config for {dataset_name}:")
    print(f"  - Dataset path: {dataset_config.get('dataset_path', 'NOT FOUND')}")
    print(f"  - Adjacency path: {dataset_config.get('adjacency_matrix_path', 'NOT FOUND')}")
    print(f"  - Expected nodes: {dataset_config.get('node_num', 'NOT FOUND')}")
    print(f"  - Expected timesteps: {dataset_config.get('time_step', 'NOT FOUND')}")
    
    # Create data manager and load the dataset
    temp_data_manager = ChaosDataManager(config)
    temp_data_manager.create_datasets(add_physics_features=config['training']['add_physics_features'])
    
    # Get and verify the loaded data matches expectations
    sample_data = temp_data_manager.get_sample_data()
    if 'temporal_data' in sample_data:
        actual_shape = sample_data['temporal_data'].shape
        actual_nodes = actual_shape[-2] if len(actual_shape) >= 3 else actual_shape[-1]
        actual_features = sample_data['temporal_data'].shape[-1]
        
        expected_nodes = dataset_config.get('node_num')
        
        print(f"Data verification for {dataset_name}:")
        print(f"  - Actual data shape: {actual_shape}")
        print(f"  - Actual nodes: {actual_nodes}")
        print(f"  - Expected nodes: {expected_nodes}")
        print(f"  - Actual features: {actual_features}")
        
        # Critical validation: ensure loaded data matches expected dataset
        if expected_nodes and actual_nodes != expected_nodes:
            raise ValueError(f"CRITICAL: Dataset mismatch detected!\n"
                           f"Expected {expected_nodes} nodes for {dataset_name}, but loaded data has {actual_nodes} nodes.\n"
                           f"This indicates the wrong dataset file is being loaded.\n"
                           f"Please check your dataset paths in the config file.\n"
                           f"Expected path: {dataset_config.get('dataset_path')}")
        
        num_nodes = actual_nodes
        actual_feature_dim = actual_features
        
    else:
        raise ValueError(f"Could not load sample data for dataset '{dataset_name}'. Please check dataset paths.")
    
    sequence_length = config['task']['his_num']
    prediction_horizon = config['task']['pred_num']
    
    print(f"Creating model with VERIFIED parameters:")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Features: {actual_feature_dim}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Prediction horizon: {prediction_horizon}")
    
    model = ChaosAwareTrafficPredictor(
        num_nodes=num_nodes,
        node_feature_dim=actual_feature_dim,
        sequence_length=sequence_length,
        hidden_dim=config['model']['hidden_dim'],
        chaos_dim=config['model']['chaos_dim'],
        output_dim=config['model']['output_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        noise_std=config['model']['noise_std'],
        prediction_horizon=prediction_horizon
    )
    
    return model.to(device)


def run_experiment(config_path, override_config=None):
    config = load_config(config_path)
    
    if override_config:
        for key, value in override_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    set_seed(config['experiment']['seed'])

    if config['experiment']['device'] == 'auto':
        device = get_device()
    else:
        device = torch.device(config['experiment']['device'])
    
    # Get test datasets from config or use default
    test_datasets = config.get('test_datasets', [config['training']['test_dataset']])
    if not isinstance(test_datasets, list):
        test_datasets = [test_datasets]
    
    all_results = {}
    
    for dataset_name in test_datasets:
        print(f"\n{'='*80}")
        print(f"TESTING ON DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Update config for current dataset
        current_config = config.copy()
        current_config['training']['test_dataset'] = dataset_name
        
        experiment_name, logger, model_saver, visualizer = setup_experiment(current_config)
        
        print(f"Starting experiment: {experiment_name}")
        print(f"Device: {device}")
        print(f"Dataset: {dataset_name}")
        print(f"Physics features: {'Enabled' if current_config['training']['add_physics_features'] else 'Disabled'}")
        print(f"Domain adaptation: {'Enabled' if current_config['few_shot']['enabled'] else 'Disabled'}")
        print("="*60)
        
        logger.log(f"Starting experiment: {experiment_name}")
        logger.log(f"Device: {device}")
        logger.log(f"Configuration: {current_config}")
        
        model = create_model(current_config, device)
        
        if dataset_name in current_config['data'] and 'node_num' in current_config['data'][dataset_name]:
            node_count = current_config['data'][dataset_name]['node_num']
        else:
            node_count = model.num_nodes
            logger.log(f"Using node count from model: {node_count}")
        
        input_shape = (
            current_config['task']['batch_size'],
            current_config['task']['his_num'],
            node_count
        )
        print_model_summary(model, input_shape)
        
        print("\nTesting model output shapes...")
        model.eval()
        with torch.no_grad():
            temp_data_manager = ChaosDataManager(current_config)
            temp_data_manager.create_datasets(add_physics_features=current_config['training']['add_physics_features'])
            dataloaders = temp_data_manager.create_dataloaders(add_physics_features=current_config['training']['add_physics_features'])
            
            sample_batch = next(iter(dataloaders['train']))
            x = sample_batch['temporal_data'].to(device)
            y = sample_batch['targets'].to(device)
            adjacency = sample_batch['adjacency'].to(device)
            
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            
            outputs, auxiliary = model(x, adjacency)
            print(f"Model output shape: {outputs.shape}")
            print(f"Expected target shape: {y.shape}")
            
            if outputs.shape[1:] == y.shape[1:]:
                print("Output and target shapes match!")
            else:
                print("Shape mismatch still exists")
                print(f"Output: {outputs.shape}, Target: {y.shape}")
        
        trainer = create_trainer(model, current_config, device, logger)
        
        print("Starting training...")
        print("="*60)
        
        with Timer() as training_timer:
            training_history = trainer.train()
        
        training_time = training_timer.elapsed()
        
        print("="*60)
        print(f"Training completed in {format_time(training_time)}")
        
        logger.log(f"Training completed in {format_time(training_time)}")
        
        if current_config['logging']['save_model_checkpoints']:
            model_path = model_saver.save_model(
                model, 
                trainer.optimizer if hasattr(trainer, 'optimizer') else trainer.meta_optimizer,
                epoch=len(training_history.get('train_loss', training_history.get('meta_loss', []))),
                metrics=training_history,
                is_best=True
            )
            logger.log(f"Model saved to: {model_path}")
            print(f"Model saved to: {model_path}")
        
        data_manager = ChaosDataManager(current_config)
        data_manager.create_datasets(add_physics_features=current_config['training']['add_physics_features'])
        dataloaders = data_manager.create_dataloaders(add_physics_features=current_config['training']['add_physics_features'])
        
        horizon_results = evaluate_horizons_improved(model, dataloaders['test'], device, 
                                                    current_config, visualizer, logger)
        
        print_horizon_results_improved(horizon_results, dataset_name, logger)
        
        final_horizon = max(horizon_results.keys(), key=lambda h: len(h))
        overall_results = horizon_results[final_horizon]
        
        print("OVERALL PERFORMANCE SUMMARY:")
        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in overall_results:
                print(f"{metric:15}: {overall_results[metric]:.6f}")
                logger.log(f"Overall {metric}: {overall_results[metric]:.6f}")
        
        summary = get_metrics_summary(overall_results)
        print(f"SUMMARY: {summary}")
        logger.log(f"Summary: {summary}")
        
        if current_config['logging']['save_metrics']:
            logger.save_metrics()
        
        if current_config['logging']['plot_training_curves']:
            logger.plot_training_curves()
        
        # Store results for this dataset
        all_results[dataset_name] = {
            'experiment_name': experiment_name,
            'training_time': training_time,
            'horizon_results': horizon_results,
            'final_metrics': overall_results,
            'training_history': training_history,
            'config': current_config
        }
    
    # Print summary of all datasets tested
    if len(test_datasets) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL DATASETS")
        print(f"{'='*80}")
        print(f"{'Dataset':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 50)
        
        for dataset_name in test_datasets:
            results = all_results[dataset_name]['final_metrics']
            mae = results.get('MAE', 0.0)
            rmse = results.get('RMSE', 0.0)
            mape = results.get('MAPE', 0.0)
            print(f"{dataset_name:<15} {mae:<10.4f} {rmse:<10.4f} {mape:<10.2f}%")
        
        print(f"{'='*80}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Chaos-Aware Traffic Flow Prediction')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--dataset', type=str, help='Override test dataset')
    parser.add_argument('--datasets', type=str, help='Multiple test datasets (comma-separated)')
    parser.add_argument('--domain_adaptation', action='store_true', help='Enable domain adaptation')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--source_epochs', type=int, help='Source epochs')
    parser.add_argument('--target_epochs', type=int, help='Target epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--physics_features', action='store_true', help='Enable physics features')
    
    args = parser.parse_args()
    
    override_config = {}
    
    if args.dataset:
        override_config['training'] = override_config.get('training', {})
        override_config['training']['test_dataset'] = args.dataset
    
    if args.domain_adaptation:
        override_config['few_shot'] = override_config.get('few_shot', {})
        override_config['few_shot']['enabled'] = True
    
    if args.epochs:
        override_config['training'] = override_config.get('training', {})
        override_config['training']['epochs'] = args.epochs
    
    if args.source_epochs:
        override_config['training'] = override_config.get('training', {})
        override_config['training']['source_epochs'] = args.source_epochs
    
    if args.target_epochs:
        override_config['training'] = override_config.get('training', {})
        override_config['training']['target_epochs'] = args.target_epochs
    
    if args.batch_size:
        override_config['task'] = override_config.get('task', {})
        override_config['task']['batch_size'] = args.batch_size
    
    if args.hidden_dim:
        override_config['model'] = override_config.get('model', {})
        override_config['model']['hidden_dim'] = args.hidden_dim
    
    if args.seed:
        override_config['experiment'] = override_config.get('experiment', {})
        override_config['experiment']['seed'] = args.seed
    
    if args.physics_features:
        override_config['training'] = override_config.get('training', {})
        override_config['training']['add_physics_features'] = True
    
    # Add support for multiple test datasets
    if args.datasets:
        override_config['test_datasets'] = args.datasets.split(',')
    
    results = run_experiment(args.config, override_config or None)
    
    print("\nExperiment completed successfully!")
    
    # Handle results for multiple datasets
    if isinstance(results, dict) and all(isinstance(v, dict) and 'horizon_results' in v for v in results.values()):
        # Multiple datasets tested
        print(f"\nTested on {len(results)} datasets:")
        for dataset_name, dataset_results in results.items():
            horizon_results = dataset_results['horizon_results']
            best_horizon = min(horizon_results.keys(), 
                              key=lambda h: horizon_results[h]['MAE'])
            best_metrics = horizon_results[best_horizon]
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Best Horizon: {best_horizon}")
            print(f"  Best MAE: {best_metrics['MAE']:.6f}")
            print(f"  Best RMSE: {best_metrics['RMSE']:.6f}")
            print(f"  Best MAPE: {best_metrics['MAPE']:.2f}%")
            print(f"  Training time: {format_time(dataset_results['training_time'])}")
        
        # Return the first dataset's results for backward compatibility
        first_dataset = list(results.keys())[0]
        return results[first_dataset]
    else:
        # Single dataset tested (backward compatibility)
        horizon_results = results['horizon_results']
        best_horizon = min(horizon_results.keys(), 
                          key=lambda h: horizon_results[h]['MAE'])
        best_metrics = horizon_results[best_horizon]
        
        print(f"Best Horizon: {best_horizon}")
        print(f"Best MAE: {best_metrics['MAE']:.6f}")
        print(f"Best RMSE: {best_metrics['RMSE']:.6f}")
        print(f"Best MAPE: {best_metrics['MAPE']:.2f}%")
        print(f"Training time: {format_time(results['training_time'])}")
        
        return results


if __name__ == "__main__":
    results = main()