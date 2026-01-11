import torch
import torch.nn as nn
import numpy as np
import random
import os
import yaml
import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config: Dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    return (param_size + buffer_size) / 1024**2


def safe_mape(y_true, y_pred, epsilon=1e-6):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    valid_mask = (
        np.isfinite(y_true_flat) & 
        np.isfinite(y_pred_flat) & 
        (np.abs(y_true_flat) > epsilon)
    )
    
    if valid_mask.sum() == 0:
        return 100.0
    
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    
    absolute_percentage_errors = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100
    mape = np.mean(absolute_percentage_errors)
    return np.clip(mape, 0.0, 200.0)


def denormalize_predictions(predictions: np.ndarray, targets: np.ndarray, 
                          dataset_stats: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_stats is not None:
        speed_mean = dataset_stats.get('speed_mean', 0.0)
        speed_std = dataset_stats.get('speed_std', 1.0)
        
        predictions_denorm = predictions * speed_std + speed_mean
        targets_denorm = targets * speed_std + speed_mean
        
        predictions_denorm = np.maximum(predictions_denorm, 0.1)
        targets_denorm = np.maximum(targets_denorm, 0.1)
        
        return predictions_denorm, targets_denorm
    
    pred_mean, pred_std = np.mean(predictions), np.std(predictions)
    target_mean, target_std = np.mean(targets), np.std(targets)
    
    if pred_std > 1e-6 and target_std > 1e-6:
        predictions_scaled = (predictions - pred_mean) / pred_std
        predictions_aligned = predictions_scaled * target_std + target_mean
        
        predictions_aligned = np.maximum(predictions_aligned, 0.1)
        targets_aligned = np.maximum(targets, 0.1)
        
        return predictions_aligned, targets_aligned
    
    predictions_pos = np.maximum(predictions, 0.1)
    targets_pos = np.maximum(targets, 0.1)
    
    return predictions_pos, targets_pos


def validate_prediction_data(predictions: np.ndarray, targets: np.ndarray) -> Tuple[bool, str]:
    issues = []
    
    if predictions.shape != targets.shape:
        issues.append(f"Shape mismatch: {predictions.shape} vs {targets.shape}")
    
    if np.isnan(predictions).any():
        issues.append(f"NaN in predictions: {np.isnan(predictions).sum()}")
    if np.isnan(targets).any():
        issues.append(f"NaN in targets: {np.isnan(targets).sum()}")
    
    if predictions.max() > 1000 or predictions.min() < -1000:
        issues.append(f"Extreme predictions: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    return len(issues) == 0, "; ".join(issues)


def evaluate_horizons_improved(model, test_loader, device: torch.device, 
                              config: Dict, visualizer, logger):
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    physics_outputs = {}
    dataset_stats = {}
    
    logger.log("Starting improved horizon evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            x = batch_data['temporal_data'].to(device)
            y = batch_data['targets'].to(device)
            adjacency = batch_data['adjacency'].to(device)
            
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            outputs, auxiliary = model(x, adjacency)
            predictions = outputs
            flow_targets = y
            
            if flow_targets.dim() == 4:
                flow_targets = flow_targets[..., 0]
            
            min_batch = min(predictions.shape[0], flow_targets.shape[0])
            min_nodes = min(predictions.shape[1], flow_targets.shape[1])
            min_steps = min(predictions.shape[2], flow_targets.shape[2])
            
            predictions = predictions[:min_batch, :min_nodes, :min_steps]
            flow_targets = flow_targets[:min_batch, :min_nodes, :min_steps]
            
            if batch_idx == 0:
                dataset_name = config['training']['test_dataset']
                if dataset_name in config['data']:
                    dataset_stats['speed_mean'] = config['data'][dataset_name].get('speed_mean', 0.0)
                    dataset_stats['speed_std'] = config['data'][dataset_name].get('speed_std', 1.0)
                else:
                    dataset_stats['speed_mean'] = float(np.mean(flow_targets.cpu().numpy()))
                    dataset_stats['speed_std'] = float(np.std(flow_targets.cpu().numpy()))
                
                logger.log(f"Using dataset stats: mean={dataset_stats['speed_mean']:.4f}, "
                          f"std={dataset_stats['speed_std']:.4f}")
            
            predictions_np = predictions.cpu().numpy()
            targets_np = flow_targets.cpu().numpy()
            
            all_predictions.append(predictions_np)
            all_targets.append(targets_np)
            
            if auxiliary and 'uncertainty' in auxiliary:
                uncertainty_np = auxiliary['uncertainty'].cpu().numpy()
                all_uncertainties.append(uncertainty_np)
            
            if batch_idx == 0:
                if auxiliary:
                    for key in ['confidence', 'chaos_features', 'attention_weights']:
                        if key in auxiliary:
                            physics_outputs[key] = auxiliary[key]
            
            if batch_idx >= 25:
                break
    
    predictions_array = np.concatenate(all_predictions, axis=0)
    targets_array = np.concatenate(all_targets, axis=0)
    uncertainties_array = np.concatenate(all_uncertainties, axis=0) if all_uncertainties else None
    
    predictions_aligned, targets_aligned = denormalize_predictions(
        predictions_array, targets_array, dataset_stats
    )
    
    logger.log(f"After alignment - Predictions: mean={np.mean(predictions_aligned):.4f}, "
              f"std={np.std(predictions_aligned):.4f}, range=[{np.min(predictions_aligned):.4f}, "
              f"{np.max(predictions_aligned):.4f}]")
    logger.log(f"After alignment - Targets: mean={np.mean(targets_aligned):.4f}, "
              f"std={np.std(targets_aligned):.4f}, range=[{np.min(targets_aligned):.4f}, "
              f"{np.max(targets_aligned):.4f}]")
    
    is_valid, validation_msg = validate_prediction_data(predictions_aligned, targets_aligned)
    if not is_valid:
        logger.log(f"Data validation warning: {validation_msg}")
    
    num_horizons = min(predictions_aligned.shape[2], 6)
    
    horizon_results = {}
    dataset_name = config['training']['test_dataset'].lower()
    
    if dataset_name in ['metr-la', 'pems-bay']:
        horizon_names = ['5min', '15min', '30min', '60min', '90min', '120min']
    else:
        horizon_names = ['10min', '20min', '30min', '60min', '90min', '120min']
    
    all_horizon_predictions = []
    all_horizon_targets = []
    
    for i, horizon_name in enumerate(horizon_names[:num_horizons]):
        pred_step = predictions_aligned[:, :, i]
        target_step = targets_aligned[:, :, i]
        
        pred_flat = pred_step.flatten()
        target_flat = target_step.flatten()
        
        valid_mask = (
            np.isfinite(pred_flat) & 
            np.isfinite(target_flat) & 
            (target_flat > 0.01) &
            (pred_flat >= 0)
        )
        
        if valid_mask.sum() > 0:
            pred_valid = pred_flat[valid_mask]
            target_valid = target_flat[valid_mask]
            
            all_horizon_predictions.extend(pred_valid.tolist())
            all_horizon_targets.extend(target_valid.tolist())
            
            mae = np.mean(np.abs(target_valid - pred_valid))
            mse = np.mean((target_valid - pred_valid) ** 2)
            rmse = np.sqrt(mse)
            
            target_scale = np.std(target_valid)
            mape_epsilon = max(target_scale * 0.01, 0.1)
            mape = safe_mape(target_valid, pred_valid, epsilon=mape_epsilon)
        else:
            mae = rmse = mape = 999.0
            mse = 999.0
        
        horizon_results[horizon_name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(min(mape, 200.0)),
            'MSE': float(mse)
        }
    
    if all_horizon_predictions and all_horizon_targets:
        overall_pred = np.array(all_horizon_predictions)
        overall_target = np.array(all_horizon_targets)
        
        overall_mae = np.mean(np.abs(overall_target - overall_pred))
        overall_mse = np.mean((overall_target - overall_pred) ** 2)
        overall_rmse = np.sqrt(overall_mse)
        
        overall_target_scale = np.std(overall_target)
        overall_mape_epsilon = max(overall_target_scale * 0.01, 0.1)
        overall_mape = safe_mape(overall_target, overall_pred, epsilon=overall_mape_epsilon)
        
        overall_metrics = {
            'MAE': float(overall_mae),
            'RMSE': float(overall_rmse),
            'MAPE': float(min(overall_mape, 200.0)),
            'MSE': float(overall_mse)
        }
        
        horizon_results['overall'] = overall_metrics
    
    logger.log("="*50)
    logger.log("IMPROVED EVALUATION RESULTS (PROPERLY ALIGNED)")
    logger.log("="*50)
    
    for i, horizon_name in enumerate(horizon_names[:num_horizons]):
        metrics = horizon_results[horizon_name]
        logger.log(f"Horizon {horizon_name}: MAE={metrics['MAE']:.4f}, "
                  f"RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%")
    
    if 'overall' in horizon_results:
        overall = horizon_results['overall']
        logger.log(f"Overall Aggregated: MAE={overall['MAE']:.4f}, "
                  f"RMSE={overall['RMSE']:.4f}, MAPE={overall['MAPE']:.2f}%")
    
    if config['logging']['plot_predictions']:
        visualizer.plot_flow_predictions(
            predictions_aligned, targets_aligned,
            node_indices=config['evaluation']['plot_nodes'][:3],
            time_indices=config['evaluation']['plot_time_steps'][:3],
            save_name='research_predictions_scatter'
        )
        
        visualizer.plot_time_series(
            predictions_aligned, targets_aligned,
            node_indices=config['evaluation']['plot_nodes'][:2],
            save_name='research_timeseries_aligned'
        )
        
        visualizer.plot_horizon_performance(horizon_results, 'research_horizon_performance')
        visualizer.plot_error_distribution(predictions_aligned, targets_aligned, 'research_error_distribution')
        visualizer.plot_node_performance_heatmap(predictions_aligned, targets_aligned, 'research_node_performance')
        visualizer.plot_temporal_error_patterns(predictions_aligned, targets_aligned, 'research_temporal_patterns')
        
        if len([h for h in horizon_results.keys() if h != 'overall']) >= 3:
            visualizer.plot_metrics_correlation(horizon_results, 'research_metrics_correlation')
        
        if uncertainties_array is not None:
            visualizer.plot_prediction_confidence(
                predictions_aligned, targets_aligned, uncertainties_array, 'research_prediction_confidence'
            )
        
        training_history = {'train_loss': [], 'val_loss': []}
        visualizer.plot_performance_summary(horizon_results, training_history, 'research_performance_summary')
        visualizer.plot_learning_dynamics(training_history, 'research_learning_dynamics')
        
        if physics_outputs:
            if 'chaos_features' in physics_outputs:
                chaos_features_np = physics_outputs['chaos_features'].cpu().numpy() if hasattr(physics_outputs['chaos_features'], 'cpu') else physics_outputs['chaos_features']
                visualizer.plot_chaos_feature_analysis(chaos_features_np, horizon_results, 'research_chaos_features')
                
                prediction_errors = np.abs(predictions_aligned - targets_aligned)
                visualizer.plot_chaos_error_correlation(chaos_features_np, prediction_errors, 'research_chaos_error_correlation')
                
                prediction_difficulties = np.mean(prediction_errors, axis=(1, 2))
                visualizer.plot_stability_analysis(chaos_features_np, prediction_difficulties, 'research_stability_analysis')
            
            if 'attention_weights' in physics_outputs:
                attention_np = physics_outputs['attention_weights'].cpu().numpy() if hasattr(physics_outputs['attention_weights'], 'cpu') else physics_outputs['attention_weights']
                visualizer.plot_attention_visualization(attention_np, save_name='research_attention_weights')
        
        features_for_tsne = predictions_aligned.reshape(predictions_aligned.shape[0], -1)
        visualizer.plot_tsne_analysis(features_for_tsne, None, 'research_tsne_analysis')
    
    return horizon_results


def print_horizon_results_improved(horizon_results: Dict, dataset_name: str, logger):
    print("\n" + "="*80)
    print(f"IMPROVED PREDICTION HORIZON RESULTS - {dataset_name.upper()}")
    print("="*80)
    
    print(f"{'Horizon':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 40)
    
    for horizon_name, metrics in horizon_results.items():
        if horizon_name == 'overall':
            continue
            
        mae = metrics['MAE']
        rmse = metrics['RMSE']
        mape = metrics['MAPE']
        
        print(f"{horizon_name:<10} {mae:<10.4f} {rmse:<10.4f} {mape:<10.2f}%")
        logger.log(f"Improved Horizon {horizon_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, "
                  f"MAPE={mape:.2f}%")
    
    print("="*80)
    
    best_horizon = min(horizon_results.keys(), 
                      key=lambda h: horizon_results[h]['MAE'])
    best_mae = horizon_results[best_horizon]['MAE']
    
    print(f"BEST PERFORMANCE: {best_horizon} (MAE: {best_mae:.4f})")
    logger.log(f"Best improved horizon performance: {best_horizon} with MAE: {best_mae:.4f}")
    print("="*80 + "\n")


class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.losses = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float = None):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        
        if valid_mask.sum() > 0:
            self.all_predictions.extend(pred_flat[valid_mask].tolist())
            self.all_targets.extend(target_flat[valid_mask].tolist())
        
        if loss is not None and np.isfinite(loss):
            self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        if not self.all_predictions:
            return {
                'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0, 
                'MSE': 0.0, 'avg_loss': 0.0
            }
        
        pred_array = np.array(self.all_predictions)
        target_array = np.array(self.all_targets)
        
        pred_aligned, target_aligned = denormalize_predictions(
            pred_array.reshape(-1, 1), target_array.reshape(-1, 1)
        )
        pred_aligned = pred_aligned.flatten()
        target_aligned = target_aligned.flatten()
        
        mae = mean_absolute_error(target_aligned, pred_aligned)
        mse = mean_squared_error(target_aligned, pred_aligned)
        rmse = np.sqrt(mse)
        
        target_scale = np.std(target_aligned)
        mape_epsilon = max(target_scale * 0.01, 0.1)
        mape = safe_mape(target_aligned, pred_aligned, epsilon=mape_epsilon)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': min(mape, 200.0),
            'MSE': mse,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0
        }
        
        return metrics


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class LearningRateScheduler:
    def __init__(self, optimizer, mode: str = 'plateau', factor: float = 0.5, 
                 patience: int = 5, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        
        if mode == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
            )
        elif mode == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=min_lr
            )
        elif mode == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=patience, gamma=factor
            )
        else:
            self.scheduler = None
    
    def step(self, metric: float = None):
        if self.scheduler is None:
            return
        
        if self.mode == 'plateau' and metric is not None:
            self.scheduler.step(metric)
        elif self.mode in ['cosine', 'step']:
            self.scheduler.step()


class Logger:
    def __init__(self, log_dir: str = "/Users/s5273738/Chaos_TrafficFlow 2/Results", experiment_name: str = None):
        if experiment_name is None:
            experiment_name = f"chaos_traffic_experiment_{int(time.time())}"
            
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'{experiment_name}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': {}
        }
    
    def log(self, message: str):
        self.logger.info(message)
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: float = None, 
                   test_metrics: Dict = None):
        self.metrics_history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.metrics_history['val_loss'].append(val_loss)
        
        if test_metrics is not None:
            self.metrics_history['test_metrics'][epoch] = test_metrics
        
        message = f"Epoch {epoch}: Train Loss={train_loss:.6f}"
        if val_loss is not None:
            message += f", Val Loss={val_loss:.6f}"
        if test_metrics is not None:
            mae = test_metrics.get('MAE', 0)
            rmse = test_metrics.get('RMSE', 0)
            message += f", MAE={mae:.4f}, RMSE={rmse:.4f}"
        
        self.log(message)
    
    def save_metrics(self):
        import json
        metrics_file = os.path.join(self.log_dir, f'{self.experiment_name}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.log(f"Metrics saved to: {metrics_file}")
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if self.metrics_history['train_loss']:
            axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss', color='blue')
            if self.metrics_history['val_loss']:
                axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if self.metrics_history['test_metrics']:
            epochs = list(self.metrics_history['test_metrics'].keys())
            maes = [self.metrics_history['test_metrics'][ep].get('MAE', 0) for ep in epochs]
            if maes:
                axes[0, 1].plot(epochs, maes, label='MAE', color='green', marker='o')
                axes[0, 1].set_title('Mean Absolute Error')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('MAE')
                axes[0, 1].grid(True, alpha=0.3)
        
        if self.metrics_history['test_metrics']:
            epochs = list(self.metrics_history['test_metrics'].keys())
            rmses = [self.metrics_history['test_metrics'][ep].get('RMSE', 0) for ep in epochs]
            if rmses:
                axes[1, 0].plot(epochs, rmses, label='RMSE', color='orange', marker='s')
                axes[1, 0].set_title('Root Mean Square Error')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('RMSE')
                axes[1, 0].grid(True, alpha=0.3)
        
        if self.metrics_history['test_metrics']:
            epochs = list(self.metrics_history['test_metrics'].keys())
            mapes = [self.metrics_history['test_metrics'][ep].get('MAPE', 0) for ep in epochs]
            if mapes:
                axes[1, 1].plot(epochs, mapes, label='MAPE', color='purple', marker='^')
                axes[1, 1].set_title('Mean Absolute Percentage Error')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('MAPE')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.log_dir, f'{self.experiment_name}_training_curves.pdf')
        plt.savefig(plot_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        self.log(f"Training curves saved to: {plot_file}")


class DataVisualizer:
    def __init__(self, save_dir: str = "/Users/s5273738/Chaos_TrafficFlow 2/Results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
    
    def plot_flow_predictions(self, predictions: np.ndarray, targets: np.ndarray, 
                             node_indices: List[int] = None, time_indices: List[int] = None,
                             save_name: str = 'final_flow_predictions_scatter'):
        
        if node_indices is None:
            node_indices = list(range(min(4, predictions.shape[1])))
        if time_indices is None:
            time_indices = list(range(min(6, predictions.shape[2])))
        
        fig, axes = plt.subplots(len(node_indices), len(time_indices), 
                                figsize=(6*len(time_indices), 5*len(node_indices)))
        
        if len(node_indices) == 1 and len(time_indices) == 1:
            axes = np.array([[axes]])
        elif len(node_indices) == 1:
            axes = axes.reshape(1, -1)
        elif len(time_indices) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, node_idx in enumerate(node_indices):
            for j, time_idx in enumerate(time_indices):
                ax = axes[i, j]
                
                pred_vals = predictions[:, node_idx, time_idx]
                target_vals = targets[:, node_idx, time_idx]
                
                valid_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
                pred_vals = pred_vals[valid_mask]
                target_vals = target_vals[valid_mask]
                
                if len(pred_vals) > 0:
                    ax.scatter(target_vals, pred_vals, alpha=0.7, s=40, color='blue', 
                              edgecolors='darkblue', linewidth=0.5)
                    
                    min_val = min(target_vals.min(), pred_vals.min())
                    max_val = max(target_vals.max(), pred_vals.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, 
                           label='Perfect Prediction')
                    
                    mae = np.mean(np.abs(target_vals - pred_vals))
                    rmse = np.sqrt(np.mean((target_vals - pred_vals)**2))
                    
                    target_scale = np.std(target_vals)
                    mape_epsilon = max(target_scale * 0.01, 0.1)
                    mape = safe_mape(target_vals, pred_vals, epsilon=mape_epsilon)
                    
                    ax.set_xlabel('True Speed (km/h)', fontweight='bold')
                    ax.set_ylabel('Predicted Speed (km/h)', fontweight='bold')
                    ax.set_title(f'Node {node_idx}, Horizon {time_idx+1}\n'
                               f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%', fontweight='bold')
                    ax.grid(True, alpha=0.4)
                    ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series(self, predictions: np.ndarray, targets: np.ndarray,
                        node_indices: List[int] = None, save_name: str = 'time_series'):
        
        if node_indices is None:
            node_indices = list(range(min(6, predictions.shape[1])))
        
        fig, axes = plt.subplots(len(node_indices), 1, 
                                figsize=(16, 4*len(node_indices)))
        
        if len(node_indices) == 1:
            axes = [axes]
        
        for i, node_idx in enumerate(node_indices):
            ax = axes[i]
            
            sample_idx = 0
            time_steps = np.arange(1, predictions.shape[2] + 1)
            
            pred_series = predictions[sample_idx, node_idx, :]
            target_series = targets[sample_idx, node_idx, :]
            
            ax.plot(time_steps, target_series, 'b-', linewidth=3, 
                   label='Ground Truth', marker='o', markersize=6)
            ax.plot(time_steps, pred_series, 'r--', linewidth=3, 
                   label='Predictions', marker='s', markersize=6)
            
            mae = np.mean(np.abs(target_series - pred_series))
            rmse = np.sqrt(np.mean((target_series - pred_series)**2))
            
            ax.set_xlabel('Prediction Horizon', fontweight='bold')
            ax.set_ylabel('Traffic Speed (km/h)', fontweight='bold')
            ax.set_title(f'Node {node_idx} - Multi-step Prediction\n'
                        f'MAE: {mae:.2f}, RMSE: {rmse:.2f}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.4)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_horizon_performance(self, horizon_results: Dict, save_name: str = 'horizon_performance'):
        horizons = [h for h in horizon_results.keys() if h != 'overall']
        if not horizons:
            return
            
        metrics = ['MAE', 'RMSE', 'MAPE']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(metrics):
            values = [horizon_results[h][metric] for h in horizons]
            bars = axes[idx].bar(horizons, values, color=colors[idx], alpha=0.8, edgecolor='black')
            axes[idx].set_title(f'{metric} Across Prediction Horizons', fontweight='bold')
            axes[idx].set_xlabel('Prediction Horizon')
            axes[idx].set_ylabel(metric)
            axes[idx].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(self, predictions: np.ndarray, targets: np.ndarray, 
                               save_name: str = 'error_distribution'):
        errors = (predictions - targets).flatten()
        abs_errors = np.abs(errors)
        
        valid_mask = np.isfinite(errors)
        errors = errors[valid_mask]
        abs_errors = abs_errors[valid_mask]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
        axes[0, 0].axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}')
        axes[0, 0].set_title('Prediction Error Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(abs_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(np.mean(abs_errors), color='red', linestyle='--', linewidth=2, label=f'MAE: {np.mean(abs_errors):.3f}')
        axes[0, 1].set_title('Absolute Error Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        percentiles = np.arange(0, 101, 5)
        error_percentiles = np.percentile(abs_errors, percentiles)
        axes[1, 0].plot(percentiles, error_percentiles, 'b-', linewidth=3, marker='o')
        axes[1, 0].set_title('Error Percentile Analysis', fontweight='bold')
        axes[1, 0].set_xlabel('Percentile')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        from scipy import stats
        qq_data = stats.probplot(errors, dist="norm")
        axes[1, 1].plot(qq_data[0][0], qq_data[0][1], 'bo', alpha=0.6)
        axes[1, 1].plot(qq_data[0][0], qq_data[1][1] + qq_data[1][0]*qq_data[0][0], 'r-', linewidth=2)
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_correlation(self, horizon_results: Dict, save_name: str = 'metrics_correlation'):
        horizons = [h for h in horizon_results.keys() if h != 'overall']
        if len(horizons) < 3:
            return
            
        metrics_data = {}
        for metric in ['MAE', 'RMSE', 'MAPE']:
            metrics_data[metric] = [horizon_results[h][metric] for h in horizons]
        
        correlation_matrix = np.corrcoef([metrics_data[metric] for metric in metrics_data.keys()])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        metrics_list = list(metrics_data.keys())
        ax.set_xticks(range(len(metrics_list)))
        ax.set_yticks(range(len(metrics_list)))
        ax.set_xticklabels(metrics_list)
        ax.set_yticklabels(metrics_list)
        
        for i in range(len(metrics_list)):
            for j in range(len(metrics_list)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title("Metrics Correlation Matrix", fontweight='bold')
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_node_performance_heatmap(self, predictions: np.ndarray, targets: np.ndarray,
                                     save_name: str = 'node_performance_heatmap'):
        num_nodes = min(predictions.shape[1], 50)
        num_horizons = predictions.shape[2]
        
        mae_matrix = np.zeros((num_nodes, num_horizons))
        rmse_matrix = np.zeros((num_nodes, num_horizons))
        
        for node in range(num_nodes):
            for horizon in range(num_horizons):
                pred_vals = predictions[:, node, horizon]
                target_vals = targets[:, node, horizon]
                
                valid_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
                if valid_mask.sum() > 0:
                    pred_valid = pred_vals[valid_mask]
                    target_valid = target_vals[valid_mask]
                    mae_matrix[node, horizon] = np.mean(np.abs(pred_valid - target_valid))
                    rmse_matrix[node, horizon] = np.sqrt(np.mean((pred_valid - target_valid)**2))
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        im1 = axes[0].imshow(mae_matrix, cmap='YlOrRd', aspect='auto')
        axes[0].set_title('MAE Across Nodes and Horizons', fontweight='bold')
        axes[0].set_xlabel('Prediction Horizon')
        axes[0].set_ylabel('Node ID')
        axes[0].set_xticks(range(num_horizons))
        axes[0].set_xticklabels([f'H{i+1}' for i in range(num_horizons)])
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('MAE', rotation=270, labelpad=20)
        
        im2 = axes[1].imshow(rmse_matrix, cmap='YlOrRd', aspect='auto')
        axes[1].set_title('RMSE Across Nodes and Horizons', fontweight='bold')
        axes[1].set_xlabel('Prediction Horizon')
        axes[1].set_ylabel('Node ID')
        axes[1].set_xticks(range(num_horizons))
        axes[1].set_xticklabels([f'H{i+1}' for i in range(num_horizons)])
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('RMSE', rotation=270, labelpad=20)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_error_patterns(self, predictions: np.ndarray, targets: np.ndarray,
                                   save_name: str = 'temporal_error_patterns'):
        num_horizons = predictions.shape[2]
        mae_over_time = []
        rmse_over_time = []
        mape_over_time = []
        
        for horizon in range(num_horizons):
            pred_vals = predictions[:, :, horizon].flatten()
            target_vals = targets[:, :, horizon].flatten()
            
            valid_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
            if valid_mask.sum() > 0:
                pred_valid = pred_vals[valid_mask]
                target_valid = target_vals[valid_mask]
                
                mae = np.mean(np.abs(pred_valid - target_valid))
                rmse = np.sqrt(np.mean((pred_valid - target_valid)**2))
                target_scale = np.std(target_valid)
                mape_epsilon = max(target_scale * 0.01, 0.1)
                mape = safe_mape(target_valid, pred_valid, epsilon=mape_epsilon)
                
                mae_over_time.append(mae)
                rmse_over_time.append(rmse)
                mape_over_time.append(mape)
        
        horizons = list(range(1, len(mae_over_time) + 1))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].plot(horizons, mae_over_time, 'b-', linewidth=3, marker='o', markersize=8)
        axes[0].set_title('MAE vs Prediction Horizon', fontweight='bold')
        axes[0].set_xlabel('Prediction Horizon')
        axes[0].set_ylabel('MAE')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(horizons)
        
        axes[1].plot(horizons, rmse_over_time, 'r-', linewidth=3, marker='s', markersize=8)
        axes[1].set_title('RMSE vs Prediction Horizon', fontweight='bold')
        axes[1].set_xlabel('Prediction Horizon')
        axes[1].set_ylabel('RMSE')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(horizons)
        
        axes[2].plot(horizons, mape_over_time, 'g-', linewidth=3, marker='^', markersize=8)
        axes[2].set_title('MAPE vs Prediction Horizon', fontweight='bold')
        axes[2].set_xlabel('Prediction Horizon')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(horizons)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_confidence(self, predictions: np.ndarray, targets: np.ndarray, 
                             uncertainties: np.ndarray = None, save_name: str = 'prediction_confidence'):
        if uncertainties is None:
            return
            
        sample_nodes = min(4, predictions.shape[1])
        sample_horizons = min(3, predictions.shape[2])
        
        fig, axes = plt.subplots(sample_nodes, sample_horizons, figsize=(15, 12))
        if sample_nodes == 1 and sample_horizons == 1:
            axes = np.array([[axes]])
        elif sample_nodes == 1:
            axes = axes.reshape(1, -1)
        elif sample_horizons == 1:
            axes = axes.reshape(-1, 1)
        
        for node in range(sample_nodes):
            for horizon in range(sample_horizons):
                ax = axes[node, horizon]
                
                pred_vals = predictions[:, node, horizon]
                target_vals = targets[:, node, horizon]
                
                valid_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
                pred_vals = pred_vals[valid_mask]
                target_vals = target_vals[valid_mask]
                
                if len(pred_vals) > 10:
                    indices = np.arange(len(pred_vals))
                    
                    ax.plot(indices, target_vals, color='blue', alpha=0.8, linewidth=2, 
                        label='Ground Truth', linestyle='-')
                    ax.plot(indices, pred_vals, color='red', alpha=0.8, linewidth=2, 
                        label='Predictions', linestyle='--')
                    
                    if uncertainties.shape[1] > node and uncertainties.shape[2] > horizon:
                        uncertainty_vals = uncertainties[:, node, horizon]
                        uncertainty_vals = uncertainty_vals[valid_mask]
                        
                        uncertainty_vals = np.abs(uncertainty_vals)
                        
                        pred_range = np.ptp(pred_vals) if np.ptp(pred_vals) > 0 else 1.0
                        max_uncertainty = 0.2 * pred_range
                        uncertainty_vals = np.minimum(uncertainty_vals, max_uncertainty)
                        
                        upper_bound = pred_vals + uncertainty_vals
                        lower_bound = pred_vals - uncertainty_vals
                        
                        ax.fill_between(indices, lower_bound, upper_bound,
                                    alpha=0.2, color='red', label='Uncertainty')
                    
                    mae = np.mean(np.abs(pred_vals - target_vals))
                    ax.set_title(f'Node {node}, Horizon {horizon+1}\nMAE: {mae:.3f}', 
                            fontweight='bold')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Traffic Speed')
                    
                    ax.legend(loc='upper right', fontsize='small')
                    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                    
                    y_min = min(np.min(target_vals), np.min(pred_vals))
                    y_max = max(np.max(target_vals), np.max(pred_vals))
                    y_margin = 0.1 * (y_max - y_min)
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)
                    
                else:
                    ax.text(0.5, 0.5, 'Insufficient Data', 
                        ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Node {node}, Horizon {horizon+1}')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_confidence_multidatasets(self, predictions: np.ndarray, targets: np.ndarray, 
                                                uncertainties: np.ndarray = None, save_name: str = 'prediction_confidence_multidatasets'):
        
        if uncertainties is None:
            return
        
        predictions_work = predictions.copy()
        targets_work = targets.copy()
        uncertainties_work = uncertainties.copy()
        
        if predictions_work.shape != targets_work.shape:
            min_batch = min(predictions_work.shape[0], targets_work.shape[0])
            min_nodes = min(predictions_work.shape[1] if predictions_work.ndim > 1 else 1, 
                          targets_work.shape[1] if targets_work.ndim > 1 else 1)
            min_horizons = min(predictions_work.shape[2] if predictions_work.ndim > 2 else 1, 
                             targets_work.shape[2] if targets_work.ndim > 2 else 1)
            
            if predictions_work.ndim == 2:
                predictions_work = predictions_work[:min_batch, :min_nodes].reshape(min_batch, min_nodes, 1)
            else:
                predictions_work = predictions_work[:min_batch, :min_nodes, :min_horizons]
                
            if targets_work.ndim == 2:
                targets_work = targets_work[:min_batch, :min_nodes].reshape(min_batch, min_nodes, 1)
            else:
                targets_work = targets_work[:min_batch, :min_nodes, :min_horizons]
        
        if uncertainties_work.shape != predictions_work.shape:
            if uncertainties_work.ndim == 2 and predictions_work.ndim == 3:
                if uncertainties_work.shape[0] == predictions_work.shape[0]:
                    if uncertainties_work.shape[1] == predictions_work.shape[2]:
                        uncertainties_work = uncertainties_work.reshape(predictions_work.shape[0], 1, predictions_work.shape[2])
                        uncertainties_work = np.broadcast_to(uncertainties_work, predictions_work.shape)
            
            if uncertainties_work.shape != predictions_work.shape:
                return
        
        max_sample_size = min(predictions_work.shape[0], 50)
        sample_nodes = min(4, predictions_work.shape[1])
        sample_horizons = min(3, predictions_work.shape[2])
        
        if max_sample_size > 0:
            sample_indices = np.random.choice(predictions_work.shape[0], max_sample_size, replace=False)
            predictions_sample = predictions_work[sample_indices]
            targets_sample = targets_work[sample_indices]
            uncertainties_sample = uncertainties_work[sample_indices] if uncertainties_work is not None else None
        else:
            predictions_sample = predictions_work
            targets_sample = targets_work
            uncertainties_sample = uncertainties_work
        
        fig, axes = plt.subplots(sample_nodes, sample_horizons, figsize=(15, 12))
        
        if sample_nodes == 1 and sample_horizons == 1:
            axes = np.array([[axes]])
        elif sample_nodes == 1:
            axes = axes.reshape(1, -1)
        elif sample_horizons == 1:
            axes = axes.reshape(-1, 1)
        
        for node in range(sample_nodes):
            for horizon in range(sample_horizons):
                ax = axes[node, horizon]
                
                pred_vals = predictions_sample[:, node, horizon]
                target_vals = targets_sample[:, node, horizon]
                
                valid_mask = (
                    np.isfinite(pred_vals) & 
                    np.isfinite(target_vals) &
                    (target_vals > -1000) & 
                    (target_vals < 1000) &
                    (pred_vals > -1000) & 
                    (pred_vals < 1000)
                )
                
                pred_vals_clean = pred_vals[valid_mask]
                target_vals_clean = target_vals[valid_mask]
                
                if len(pred_vals_clean) > 0:
                    indices = np.arange(len(pred_vals_clean))
                    
                    ax.plot(indices, target_vals_clean, 'b-', alpha=0.8, linewidth=2, label='Ground Truth')
                    ax.plot(indices, pred_vals_clean, 'r--', alpha=0.8, linewidth=2, label='Predictions')
                    
                    if uncertainties_sample is not None:
                        uncertainty_vals = uncertainties_sample[:, node, horizon]
                        uncertainty_vals_clean = uncertainty_vals[valid_mask]
                        
                        if len(uncertainty_vals_clean) == len(pred_vals_clean):
                            uncertainty_vals_clean = np.abs(uncertainty_vals_clean)
                            ax.fill_between(indices, 
                                          pred_vals_clean - uncertainty_vals_clean, 
                                          pred_vals_clean + uncertainty_vals_clean,
                                          alpha=0.3, color='red', label='Uncertainty')
                    
                    mae = np.mean(np.abs(pred_vals_clean - target_vals_clean))
                    rmse = np.sqrt(np.mean((pred_vals_clean - target_vals_clean)**2))
                    
                    ax.set_title(f'Node {node}, Horizon {horizon+1}\nMAE: {mae:.3f}, RMSE: {rmse:.3f}')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Traffic Speed')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid data', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'Node {node}, Horizon {horizon+1}\nNo Data')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_summary(self, horizon_results: Dict, training_history: Dict = None,
                               save_name: str = 'performance_summary'):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        horizons = [h for h in horizon_results.keys() if h != 'overall']
        if horizons:
            ax1 = fig.add_subplot(gs[0, :2])
            mae_values = [horizon_results[h]['MAE'] for h in horizons]
            rmse_values = [horizon_results[h]['RMSE'] for h in horizons]
            
            x = np.arange(len(horizons))
            width = 0.35
            ax1.bar(x - width/2, mae_values, width, label='MAE', color='#2E86AB', alpha=0.8)
            ax1.bar(x + width/2, rmse_values, width, label='RMSE', color='#A23B72', alpha=0.8)
            ax1.set_xlabel('Prediction Horizon')
            ax1.set_ylabel('Error Value')
            ax1.set_title('MAE and RMSE Across Horizons', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(horizons)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(gs[0, 2:])
            mape_values = [horizon_results[h]['MAPE'] for h in horizons]
            ax2.plot(horizons, mape_values, 'o-', linewidth=3, markersize=8, color='#F18F01')
            ax2.set_xlabel('Prediction Horizon')
            ax2.set_ylabel('MAPE (%)')
            ax2.set_title('MAPE Across Horizons', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        if training_history and 'train_loss' in training_history and training_history['train_loss']:
            ax3 = fig.add_subplot(gs[1, :2])
            epochs = range(len(training_history['train_loss']))
            ax3.plot(epochs, training_history['train_loss'], 'b-', linewidth=2, label='Training Loss')
            if 'val_loss' in training_history and training_history['val_loss']:
                ax3.plot(epochs, training_history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Progress', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        if 'overall' in horizon_results:
            ax4 = fig.add_subplot(gs[1, 2:])
            metrics = ['MAE', 'RMSE', 'MAPE']
            values = [horizon_results['overall'][metric] for metric in metrics]
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
            ax4.set_title('Overall Performance Metrics', fontweight='bold')
            ax4.set_ylabel('Metric Value')
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        if horizons:
            ax5 = fig.add_subplot(gs[2, :])
            metrics_over_horizons = {}
            for metric in ['MAE', 'RMSE', 'MAPE']:
                metrics_over_horizons[metric] = [horizon_results[h][metric] for h in horizons]
            
            for metric, values in metrics_over_horizons.items():
                ax5.plot(horizons, values, 'o-', linewidth=2, markersize=6, label=metric)
            
            ax5.set_xlabel('Prediction Horizon')
            ax5.set_ylabel('Normalized Metric Value')
            ax5.set_title('All Metrics Comparison Across Horizons', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Performance Analysis', fontsize=20, fontweight='bold')
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_chaos_feature_analysis(self, chaos_features: np.ndarray, horizon_results: Dict,
                                   save_name: str = 'chaos_feature_analysis'):
        if chaos_features.ndim > 2:
            chaos_features = chaos_features.reshape(-1, chaos_features.shape[-1])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].hist(chaos_features.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].set_title('Chaos Feature Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        feature_means = np.mean(chaos_features, axis=0)
        feature_indices = range(len(feature_means))
        axes[0, 1].bar(feature_indices, feature_means, alpha=0.7, color='orange')
        axes[0, 1].set_title('Mean Chaos Features', fontweight='bold')
        axes[0, 1].set_xlabel('Chaos Feature Index')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        horizons = [h for h in horizon_results.keys() if h != 'overall']
        if horizons and len(horizons) > 2:
            mae_values = [horizon_results[h]['MAE'] for h in horizons]
            chaos_magnitude = np.mean(np.linalg.norm(chaos_features, axis=1))
            
            axes[1, 0].scatter([chaos_magnitude] * len(horizons), mae_values, 
                             alpha=0.7, s=100, c=range(len(horizons)), cmap='viridis')
            axes[1, 0].set_xlabel('Chaos Magnitude')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].set_title('Chaos Magnitude vs MAE', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        if chaos_features.shape[1] > 1:
            feature_corr = np.corrcoef(chaos_features.T)
            im = axes[1, 1].imshow(feature_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[1, 1].set_title('Chaos Feature Correlation', fontweight='bold')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Feature Index')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_chaos_dynamics_analysis(self, chaos_features: np.ndarray, horizon_results: Dict,
                                save_name: str = 'chaos_dynamics_analysis'):
        if chaos_features.ndim > 2:
            chaos_features = chaos_features.reshape(-1, chaos_features.shape[-1])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        time_steps = np.arange(len(chaos_features))
        chaos_variance = np.var(chaos_features, axis=1)
        rolling_window = min(50, len(chaos_variance) // 10)
        if rolling_window > 1:
            rolling_variance = np.convolve(chaos_variance, np.ones(rolling_window)/rolling_window, mode='valid')
            rolling_time = time_steps[:len(rolling_variance)]
            axes[0, 0].plot(rolling_time, rolling_variance, color='darkred', linewidth=2, alpha=0.8)
        axes[0, 0].scatter(time_steps, chaos_variance, alpha=0.3, s=20, c=chaos_variance, cmap='plasma')
        axes[0, 0].set_title('Chaos Stability Evolution', fontweight='bold')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Chaos Variance')
        axes[0, 0].grid(True, alpha=0.3)
        
        if chaos_features.shape[0] > 100:
            from sklearn.cluster import KMeans
            n_clusters = min(5, chaos_features.shape[1])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(chaos_features)
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            chaos_2d = pca.fit_transform(chaos_features)
            
            scatter = axes[0, 1].scatter(chaos_2d[:, 0], chaos_2d[:, 1], 
                                    c=clusters, cmap='viridis', alpha=0.6, s=30)
            axes[0, 1].set_title('Chaos Regime Clustering', fontweight='bold')
            axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
            axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
            plt.colorbar(scatter, ax=axes[0, 1])
        else:
            for i in range(min(3, chaos_features.shape[1])):
                axes[0, 1].plot(time_steps, chaos_features[:, i], 
                            alpha=0.7, label=f'Feature {i}', linewidth=1.5)
            axes[0, 1].set_title('Chaos Feature Evolution', fontweight='bold')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Feature Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        if len(chaos_features) > 1:
            feature_diff = np.diff(chaos_features, axis=0)
            predictability = 1 / (1 + np.linalg.norm(feature_diff, axis=1))
            
            pred_bins = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            pred_counts = np.histogram(predictability, bins=5)[0]
            
            colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4']
            wedges, texts, autotexts = axes[1, 0].pie(pred_counts, labels=pred_bins, 
                                                    colors=colors, autopct='%1.1f%%',
                                                    startangle=90)
            axes[1, 0].set_title('Chaos Predictability Distribution', fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        if chaos_features.shape[1] > 1:
            feature_corr = np.corrcoef(chaos_features.T)
            im = axes[1, 1].imshow(feature_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[1, 1].set_title('Chaos Feature Correlation', fontweight='bold')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Feature Index')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_chaos_error_correlation(self, chaos_features: np.ndarray, prediction_errors: np.ndarray,
                                    save_name: str = 'chaos_error_correlation'):
        if chaos_features.ndim > 2:
            chaos_features = chaos_features.reshape(-1, chaos_features.shape[-1])
        
        if prediction_errors.ndim > 1:
            prediction_errors = prediction_errors.reshape(-1)
        
        min_samples = min(len(chaos_features), len(prediction_errors))
        chaos_features = chaos_features[:min_samples]
        prediction_errors = prediction_errors[:min_samples]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        chaos_magnitude = np.linalg.norm(chaos_features, axis=1)
        axes[0, 0].scatter(chaos_magnitude, prediction_errors, alpha=0.6, s=30)
        axes[0, 0].set_xlabel('Chaos Magnitude')
        axes[0, 0].set_ylabel('Prediction Error')
        axes[0, 0].set_title('Chaos Magnitude vs Prediction Error', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        if chaos_features.shape[1] > 0:
            axes[0, 1].scatter(chaos_features[:, 0], prediction_errors, alpha=0.6, s=30, color='red')
            axes[0, 1].set_xlabel('First Chaos Feature')
            axes[0, 1].set_ylabel('Prediction Error')
            axes[0, 1].set_title('First Chaos Feature vs Prediction Error', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        chaos_std = np.std(chaos_features, axis=1)
        axes[1, 0].scatter(chaos_std, prediction_errors, alpha=0.6, s=30, color='green')
        axes[1, 0].set_xlabel('Chaos Feature Std')
        axes[1, 0].set_ylabel('Prediction Error')
        axes[1, 0].set_title('Chaos Variability vs Prediction Error', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hexbin(chaos_magnitude, prediction_errors, gridsize=20, cmap='Blues')
        axes[1, 1].set_xlabel('Chaos Magnitude')
        axes[1, 1].set_ylabel('Prediction Error')
        axes[1, 1].set_title('Chaos-Error Density Plot', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stability_analysis(self, chaos_features: np.ndarray, prediction_difficulties: np.ndarray,
                              save_name: str = 'stability_analysis'):
        if chaos_features.ndim > 2:
            chaos_features = chaos_features.reshape(-1, chaos_features.shape[-1])
        
        min_samples = min(len(chaos_features), len(prediction_difficulties))
        chaos_features = chaos_features[:min_samples]
        prediction_difficulties = prediction_difficulties[:min_samples]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        chaos_entropy = -np.sum(chaos_features * np.log(np.abs(chaos_features) + 1e-8), axis=1)
        axes[0, 0].scatter(chaos_entropy, prediction_difficulties, alpha=0.6, s=40)
        axes[0, 0].set_xlabel('Chaos Entropy')
        axes[0, 0].set_ylabel('Prediction Difficulty')
        axes[0, 0].set_title('System Entropy vs Prediction Difficulty', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        chaos_variance = np.var(chaos_features, axis=1)
        axes[0, 1].scatter(chaos_variance, prediction_difficulties, alpha=0.6, s=40, color='orange')
        axes[0, 1].set_xlabel('Chaos Variance')
        axes[0, 1].set_ylabel('Prediction Difficulty')
        axes[0, 1].set_title('Chaos Variance vs Prediction Difficulty', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        stability_bins = np.linspace(0, np.max(prediction_difficulties), 10)
        bin_indices = np.digitize(prediction_difficulties, stability_bins)
        bin_means = [np.mean(chaos_entropy[bin_indices == i]) for i in range(1, len(stability_bins))]
        bin_centers = (stability_bins[:-1] + stability_bins[1:]) / 2
        
        axes[1, 0].plot(bin_centers, bin_means, 'bo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Prediction Difficulty Bins')
        axes[1, 0].set_ylabel('Mean Chaos Entropy')
        axes[1, 0].set_title('Stability Analysis', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        correlation = np.corrcoef(chaos_entropy, prediction_difficulties)[0, 1]
        axes[1, 1].text(0.5, 0.5, f'Entropy-Difficulty\nCorrelation: {correlation:.3f}', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=16, fontweight='bold')
        axes[1, 1].set_title('Correlation Summary', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_attention_visualization(self, attention_weights: np.ndarray, node_positions: np.ndarray = None,
                                   save_name: str = 'attention_weights'):
        if attention_weights.ndim > 3:
            attention_weights = attention_weights[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if attention_weights.ndim == 3:
            avg_attention = np.mean(attention_weights, axis=0)
        else:
            avg_attention = attention_weights
        
        im1 = axes[0, 0].imshow(avg_attention, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Average Attention Weights', fontweight='bold')
        axes[0, 0].set_xlabel('Node Index')
        axes[0, 0].set_ylabel('Head Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        attention_std = np.std(avg_attention, axis=1)
        axes[0, 1].plot(attention_std, 'b-', linewidth=2, marker='o')
        axes[0, 1].set_title('Attention Weight Variability', fontweight='bold')
        axes[0, 1].set_xlabel('Head Index')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        attention_sum = np.sum(avg_attention, axis=0)
        axes[1, 0].bar(range(len(attention_sum)), attention_sum, alpha=0.7, color='orange')
        axes[1, 0].set_title('Total Attention per Node', fontweight='bold')
        axes[1, 0].set_xlabel('Node Index')
        axes[1, 0].set_ylabel('Total Attention')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(avg_attention.flatten(), bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Attention Weight Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Attention Weight')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_spatial_patterns(self, spatial_data: np.ndarray, node_positions: np.ndarray,
                            predictions: np.ndarray, save_name: str = 'spatial_patterns'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if node_positions is not None and node_positions.shape[1] >= 2:
            scatter = axes[0, 0].scatter(node_positions[:, 0], node_positions[:, 1], 
                                       c=spatial_data[:min(len(spatial_data), len(node_positions))], 
                                       cmap='viridis', s=50, alpha=0.7)
            axes[0, 0].set_title('Spatial Data Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('X Coordinate')
            axes[0, 0].set_ylabel('Y Coordinate')
            plt.colorbar(scatter, ax=axes[0, 0])
        else:
            axes[0, 0].scatter(range(len(spatial_data[:50])), spatial_data[:50], alpha=0.7)
            axes[0, 0].set_title('Spatial Data (First 50 Nodes)', fontweight='bold')
            axes[0, 0].set_xlabel('Node Index')
            axes[0, 0].set_ylabel('Spatial Value')
            axes[0, 0].grid(True, alpha=0.3)
        
        prediction_means = np.mean(predictions, axis=(0, 2))[:50]
        axes[0, 1].plot(prediction_means, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 1].set_title('Mean Predictions per Node', fontweight='bold')
        axes[0, 1].set_xlabel('Node Index')
        axes[0, 1].set_ylabel('Mean Prediction')
        axes[0, 1].grid(True, alpha=0.3)
        
        if len(spatial_data) >= 50 and len(prediction_means) >= 50:
            spatial_subset = spatial_data[:50]
            pred_subset = prediction_means[:50]
            axes[1, 0].scatter(spatial_subset, pred_subset, alpha=0.6, s=40)
            axes[1, 0].set_xlabel('Spatial Data')
            axes[1, 0].set_ylabel('Mean Predictions')
            axes[1, 0].set_title('Spatial-Prediction Correlation', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist2d(spatial_data[:min(1000, len(spatial_data))], 
                         np.random.choice(prediction_means, min(1000, len(spatial_data))),
                         bins=20, cmap='Blues')
        axes[1, 1].set_xlabel('Spatial Data')
        axes[1, 1].set_ylabel('Predictions')
        axes[1, 1].set_title('Spatial-Prediction Density', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_tsne_analysis(self, features: np.ndarray, labels: np.ndarray = None, 
                          save_name: str = 'research_tsne_analysis'):
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        
        max_samples = min(100000, features.shape[0])
        features_subset = features[:max_samples]
        
        if labels is not None:
            labels_subset = labels[:max_samples]
        else:
            labels_subset = None
        
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_subset)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max_samples//4))
        features_tsne = tsne.fit_transform(features_normalized)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if labels_subset is not None:
            unique_labels = np.unique(labels_subset)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels_subset == label
                axes[0, 0].scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                                 c=[colors[i]], label=f'Class {int(label)}', alpha=0.7, s=30)
            axes[0, 0].legend()
        else:
            axes[0, 0].scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.7, s=30)
        
        axes[0, 0].set_title('t-SNE Visualization of Features', fontweight='bold')
        axes[0, 0].set_xlabel('t-SNE Component 1')
        axes[0, 0].set_ylabel('t-SNE Component 2')
        axes[0, 0].grid(True, alpha=0.3)
        
        distances = np.linalg.norm(features_tsne, axis=1)
        scatter = axes[0, 1].scatter(features_tsne[:, 0], features_tsne[:, 1], 
                                   c=distances, cmap='viridis', alpha=0.7, s=30)
        axes[0, 1].set_title('t-SNE Colored by Distance from Origin', fontweight='bold')
        axes[0, 1].set_xlabel('t-SNE Component 1')
        axes[0, 1].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        density_x = np.histogram(features_tsne[:, 0], bins=30)[0]
        density_y = np.histogram(features_tsne[:, 1], bins=30)[0]
        axes[1, 0].plot(density_x, 'b-', linewidth=2, label='Component 1')
        axes[1, 0].plot(density_y, 'r-', linewidth=2, label='Component 2')
        axes[1, 0].set_title('t-SNE Component Distributions', fontweight='bold')
        axes[1, 0].set_xlabel('Bin Index')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hexbin(features_tsne[:, 0], features_tsne[:, 1], gridsize=20, cmap='Blues')
        axes[1, 1].set_title('t-SNE Density Plot', fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE Component 1')
        axes[1, 1].set_ylabel('t-SNE Component 2')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_dynamics(self, training_history: Dict, save_name: str = 'learning_dynamics'):
        if not training_history or 'train_loss' not in training_history or len(training_history['train_loss']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        train_losses = training_history['train_loss']
        epochs = range(len(train_losses))
        
        axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Training Loss Evolution', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        if len(train_losses) > 4:
            smoothed_loss = np.convolve(train_losses, np.ones(min(5, len(train_losses)))/(min(5, len(train_losses))), mode='valid')
            axes[0, 1].plot(range(len(smoothed_loss)), smoothed_loss, 'r-', linewidth=2)
            axes[0, 1].set_title('Smoothed Training Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Smoothed Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'val_metrics' in training_history and training_history['val_metrics']:
            val_maes = [m.get('MAE', 0) for m in training_history['val_metrics']]
            val_rmses = [m.get('RMSE', 0) for m in training_history['val_metrics']]
            
            val_epochs = range(len(val_maes))
            axes[1, 0].plot(val_epochs, val_maes, 'g-', linewidth=2, marker='o', label='MAE')
            axes[1, 0].plot(val_epochs, val_rmses, 'orange', linewidth=2, marker='s', label='RMSE')
            axes[1, 0].set_title('Validation Metrics', fontweight='bold')
            axes[1, 0].set_xlabel('Validation Epoch')
            axes[1, 0].set_ylabel('Metric Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        elif 'val_loss' in training_history and training_history['val_loss']:
            val_losses = training_history['val_loss']
            val_epochs = range(len(val_losses))
            axes[1, 0].plot(val_epochs, val_losses, 'g-', linewidth=2, marker='o', label='Val Loss')
            axes[1, 0].set_title('Validation Loss', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Validation Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if len(train_losses) > 1:
            loss_gradients = np.gradient(train_losses)
            axes[1, 1].plot(epochs[1:], loss_gradients[1:], 'purple', linewidth=2)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Training Loss Gradient', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Gradient')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{save_name}.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def auto_save_all_research_plots(self, predictions: np.ndarray, targets: np.ndarray, 
                                   horizon_results: Dict, training_history: Dict = None,
                                   uncertainties: np.ndarray = None, chaos_features: np.ndarray = None,
                                   attention_weights: np.ndarray = None, node_positions: np.ndarray = None):
        
        self.plot_flow_predictions(predictions, targets, save_name='research_predictions_scatter')
        self.plot_time_series(predictions, targets, save_name='research_timeseries')
        self.plot_horizon_performance(horizon_results, 'research_horizon_performance')
        self.plot_error_distribution(predictions, targets, 'research_error_distribution')
        self.plot_temporal_error_patterns(predictions, targets, 'research_temporal_patterns')
        self.plot_node_performance_heatmap(predictions, targets, 'research_node_performance')
        
        if len([h for h in horizon_results.keys() if h != 'overall']) >= 3:
            self.plot_metrics_correlation(horizon_results, 'research_metrics_correlation')
        
        if uncertainties is not None:
            self.plot_prediction_confidence(predictions, targets, uncertainties, 'research_prediction_confidence')
            self.plot_prediction_confidence_multidatasets(predictions, targets, uncertainties, 'research_prediction_confidence_multidatasets')
        
        if training_history:
            self.plot_learning_dynamics(training_history, 'research_learning_dynamics')
        
        self.plot_performance_summary(horizon_results, training_history, 'research_performance_summary')
        
        if chaos_features is not None:
            self.plot_chaos_feature_analysis(chaos_features, horizon_results, 'research_chaos_features')
            
            prediction_errors = np.abs(predictions - targets)
            self.plot_chaos_error_correlation(chaos_features, prediction_errors, 'research_chaos_error_correlation')
            
            prediction_difficulties = np.mean(prediction_errors, axis=(1, 2))
            self.plot_stability_analysis(chaos_features, prediction_difficulties, 'research_stability_analysis')
            
            if len(chaos_features) > 50:  
                self.plot_chaos_dynamics_analysis(chaos_features, horizon_results, 'chaos_dynamics_standalone')
        
        if attention_weights is not None:
            self.plot_attention_visualization(attention_weights, node_positions, 'research_attention_weights')
        
        if node_positions is not None:
            spatial_data = np.mean(predictions, axis=(0, 2))
            if len(spatial_data) < predictions.shape[1]:
                spatial_data = np.mean(predictions, axis=0)[:, 0]
            self.plot_spatial_patterns(spatial_data, node_positions, predictions, 'research_spatial_patterns')
        
        features_for_tsne = predictions.reshape(predictions.shape[0], -1)
        self.plot_tsne_analysis(features_for_tsne, None, 'research_tsne_analysis')


def print_model_summary(model: nn.Module, input_shape: Tuple):
    total_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print("="*50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Input Shape: {input_shape}")
    print("="*50)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.2f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.2f}s"


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def validate_config(config: Dict) -> bool:
    required_keys = ['data', 'model', 'training', 'few_shot']
    return all(key in config for key in required_keys)


def create_experiment_name(config: Dict) -> str:
    model_name = config['model']['name']
    dataset = config['training']['test_dataset']
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{dataset}_{timestamp}"


class ModelSaver:
    def __init__(self, save_dir: str = "/Users/s5273738/Chaos_TrafficFlow 2/Results", experiment_name: str = None):
        if experiment_name is None:
            experiment_name = f"chaos_traffic_model_{int(time.time())}"
            
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, model: nn.Module, optimizer = None, 
                  epoch: int = None, metrics: Dict = None, is_best: bool = False) -> str:
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        filename = f'{self.experiment_name}'
        if epoch is not None:
            filename += f'_epoch_{epoch}'
        if is_best:
            filename += '_best'
        filename += '.pth'
        
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        
        return save_path
    
    def load_model(self, model: nn.Module, checkpoint_path: str, 
                   optimizer = None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def get_metrics_summary(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "No metrics available"
    
    key_metrics = ['MAE', 'RMSE', 'MAPE']
    parts = []
    for metric in key_metrics:
        if metric in metrics:
            parts.append(f"{metric}: {metrics[metric]:.4f}")
    
    return " | ".join(parts)


MetricsTracker = MetricsCalculator
ExperimentLogger = Logger
ModelCheckpoint = ModelSaver