import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List

from models.model import ChaosAwareTrafficPredictor
from datasets import ChaosDataManager
from utils import MetricsCalculator, EarlyStopping, LearningRateScheduler, Timer, format_time


class DomainAdaptationTrainer:
    def __init__(self, model: ChaosAwareTrafficPredictor, config: Dict, device: torch.device, logger):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        
        self.adaptation_steps = int(config.get('few_shot', {}).get('adaptation_steps', 5))
        
        self.criterion = self._create_loss_function()
        
        self.data_manager = ChaosDataManager(config)
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['training'].get('source_lr', 0.0005)),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        self.scheduler = LearningRateScheduler(
            self.meta_optimizer,
            mode=config['training']['scheduler']['type'],
            factor=float(config['training']['scheduler']['factor']),
            patience=int(config['training']['scheduler']['patience']),
            min_lr=float(config['training']['scheduler']['min_lr'])
        )
        
        self.metrics_calculator = MetricsCalculator()
        
        self.early_stopping = EarlyStopping(
            patience=int(config.get('training', {}).get('early_stopping', {}).get('patience', 15)),
            min_delta=float(config.get('training', {}).get('early_stopping', {}).get('min_delta', 1e-5)),
            restore_best=True
        )
        self.timer = Timer()

    def _create_loss_function(self):
        return nn.MSELoss()
    
    def _align_shapes_for_metrics(self, outputs, targets, batch_idx=0):
        outputs_aligned = outputs
        targets_aligned = targets
        
        if outputs.shape != targets.shape:
            if outputs.dim() == 3 and targets.dim() == 2:
                targets_aligned = targets.unsqueeze(1).expand(-1, outputs.size(1), -1)
            elif outputs.dim() == 3 and targets.dim() == 3:
                if outputs.shape[1] == 1 and targets.shape[2] == 1:
                    outputs_aligned = outputs.squeeze(1)
                    targets_aligned = targets.squeeze(-1)
                elif outputs.shape[1] != targets.shape[1] and outputs.shape[2] == targets.shape[1]:
                    outputs_aligned = outputs.transpose(1, 2)
                elif outputs.shape[1] == targets.shape[2] and outputs.shape[2] == targets.shape[1]:
                    targets_aligned = targets.transpose(1, 2)
            elif outputs.dim() == 2 and targets.dim() == 3:
                targets_aligned = targets[:, :, 0] if targets.shape[2] > 0 else targets.squeeze(-1)
            elif outputs.dim() == 2 and targets.dim() == 2:
                if outputs.shape[1] != targets.shape[1]:
                    min_features = min(outputs.shape[1], targets.shape[1])
                    outputs_aligned = outputs[:, :min_features]
                    targets_aligned = targets[:, :min_features]
            else:
                batch_size = outputs.size(0)
                outputs_flat = outputs.view(batch_size, -1)
                targets_flat = targets.view(batch_size, -1)
                
                min_features = min(outputs_flat.size(1), targets_flat.size(1))
                outputs_aligned = outputs_flat[:, :min_features]
                targets_aligned = targets_flat[:, :min_features]
        
        return outputs_aligned, targets_aligned
        
    def inner_adaptation(self, support_data, support_adjacency) -> nn.Module:
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=float(self.config.get('few_shot', {}).get('inner_lr', 0.001))
        )
        
        adapted_model.train()
        for step in range(self.adaptation_steps):
            inner_optimizer.zero_grad()
            
            x_support = support_data['temporal_data'].to(self.device)
            y_support = support_data['targets'].to(self.device)
            adjacency = support_adjacency.to(self.device)
            
            if x_support.dim() == 3:
                x_support = x_support.unsqueeze(1)
            
            outputs, auxiliary = adapted_model(x_support, adjacency)
            
            outputs_aligned, targets_aligned = self._align_shapes_for_metrics(outputs, y_support, step)
            loss = self.criterion(outputs_aligned, targets_aligned)
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 
                                             float(self.config.get('few_shot', {}).get('gradient_clip_inner', 1.0)))
                inner_optimizer.step()
        
        return adapted_model
    
    def meta_update(self, source_loader, target_loader):
        meta_losses = []
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        source_batch = next(source_iter)
        target_batch = next(target_iter)
        
        adapted_model = self.inner_adaptation(source_batch, source_batch['adjacency'])
        adapted_model.eval()
        
        x_query = target_batch['temporal_data'].to(self.device)
        y_query = target_batch['targets'].to(self.device)
        query_adjacency = target_batch['adjacency'].to(self.device)
        
        if x_query.dim() == 3:
            x_query = x_query.unsqueeze(1)
        
        outputs, auxiliary = adapted_model(x_query, query_adjacency)
        
        outputs_aligned, targets_aligned = self._align_shapes_for_metrics(outputs, y_query)
        loss = self.criterion(outputs_aligned, targets_aligned)
        meta_losses.append(loss)
        
        meta_loss = torch.stack(meta_losses).mean()
        
        self.meta_optimizer.zero_grad()
        if torch.isfinite(meta_loss):
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         float(self.config.get('few_shot', {}).get('gradient_clip_outer', 1.0)))
            self.meta_optimizer.step()
        
        return {'meta_loss': meta_loss.item()}
    
    def evaluate_on_target(self, target_loader, num_batches: int = 5):
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(target_loader):
                if batch_idx >= num_batches:
                    break
                
                x = batch_data['temporal_data'].to(self.device)
                y = batch_data['targets'].to(self.device)
                adjacency = batch_data['adjacency'].to(self.device)
                
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                
                outputs, auxiliary = self.model(x, adjacency)
                
                outputs_aligned, targets_aligned = self._align_shapes_for_metrics(outputs, y, batch_idx)
                
                loss = self.criterion(outputs_aligned, targets_aligned)
                
                if outputs_aligned.shape == targets_aligned.shape and outputs_aligned.numel() > 0:
                    self.metrics_calculator.update(outputs_aligned, targets_aligned, loss.item())
                    total_loss += loss.item()
                    valid_batches += 1
                else:
                    print(f"Skipping batch {batch_idx}: final shapes {outputs_aligned.shape} vs {targets_aligned.shape}")
       
        if valid_batches == 0:
            print("WARNING: No valid batches for evaluation!")
            return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0, 'episode_loss': 0.0}
        
        metrics = self.metrics_calculator.compute_metrics()
        metrics['episode_loss'] = total_loss / valid_batches
        
        return metrics
    
    def train(self):
        self.timer.start()
        self.logger.log("Starting enhanced domain adaptation training with ChaosDataManager")
        
        self.logger.log("Creating datasets with ChaosDataManager...")
        self.data_manager.create_datasets(add_physics_features=self.config['training']['add_physics_features'])
        
        dataloaders = self.data_manager.create_dataloaders(
            add_physics_features=self.config['training']['add_physics_features']
        )
        
        source_loader = dataloaders['train']
        target_loader = dataloaders['val']
        
        dataset_info = self.data_manager.get_dataset_info()
        self.logger.log("Dataset Information:")
        for split, info in dataset_info.items():
            self.logger.log(f"  {split}: {info['num_samples']} samples, "
                          f"{info['num_nodes']} nodes, {info['num_features']} features, "
                          f"Physics features: {info['physics_features_enabled']}")
        
        training_history = {
            'meta_loss': [],
            'val_metrics': []
        }
        
        source_epochs = min(int(self.config['training']['source_epochs']), 300)
        self.logger.log(f"PHASE 1: Source domain training for {source_epochs} epochs")
        
        print("="*60)
        print(f"PHASE 1: SOURCE DOMAIN TRAINING ({source_epochs} epochs)")
        print(f"Dataset: {self.config['training']['test_dataset']}")
        print(f"Physics features: {self.config['training']['add_physics_features']}")
        print("="*60)
        
        standard_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config['training']['source_lr']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        for epoch in range(source_epochs):
            epoch_losses = []
            
            for batch_idx, batch_data in enumerate(source_loader):
                self.model.train()
                standard_optimizer.zero_grad()
                
                x = batch_data['temporal_data'].to(self.device)
                y = batch_data['targets'].to(self.device)
                adjacency = batch_data['adjacency'].to(self.device)
                
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                
                outputs, auxiliary = self.model(x, adjacency)
                
                outputs_aligned, targets_aligned = self._align_shapes_for_metrics(
                    outputs, y, -1
                )
                loss = self.criterion(outputs_aligned, targets_aligned)
                
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 float(self.config['training'].get('gradient_clip', 1.0)))
                    standard_optimizer.step()
                    epoch_losses.append(loss.item())
                
                if batch_idx >= 20:
                    break
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            training_history['meta_loss'].append(avg_loss)
            
            if epoch % 10 == 0 or epoch == source_epochs - 1:
                val_metrics = self.evaluate_on_target(target_loader, 5)
                training_history['val_metrics'].append(val_metrics)
                
                mae = val_metrics.get('MAE', 0.0)
                rmse = val_metrics.get('RMSE', 0.0)
                
                print(f"Source Epoch {epoch:3d}/{source_epochs} | "
                      f"Loss: {avg_loss:.6f} | "
                      f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")
                
                self.logger.log_metrics(epoch, avg_loss, val_metrics['episode_loss'], val_metrics)
                
                self.scheduler.step(val_metrics['episode_loss'])
                
                if self.early_stopping(val_metrics['episode_loss'], self.model):
                    elapsed_time = format_time(self.timer.elapsed())
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Training time: {elapsed_time}")
                    self.logger.log(f"Early stopping at epoch {epoch}. Time: {elapsed_time}")
                    break
            else:
                print(f"Source Epoch {epoch:3d}/{source_epochs} | Loss: {avg_loss:.6f}")
        
        target_epochs = min(int(self.config['training']['target_epochs']), 300)
        target_lr = float(self.config['training'].get('target_lr', 0.0002))
        
        self.logger.log(f"PHASE 2: Target domain adaptation for {target_epochs} epochs")
        print(f"\n" + "="*60)
        print(f"PHASE 2: TARGET DOMAIN ADAPTATION ({target_epochs} epochs, lr={target_lr})")
        print("="*60)
        
        if target_epochs > 0:
            target_optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=target_lr,
                weight_decay=float(self.config['training']['weight_decay'])
            )
            
            for epoch in range(target_epochs):
                epoch_loss = 0.0
                num_updates = 0
                
                for batch_idx, batch_data in enumerate(target_loader):
                    target_optimizer.zero_grad()
                    
                    x = batch_data['temporal_data'].to(self.device)
                    y = batch_data['targets'].to(self.device)
                    adjacency = batch_data['adjacency'].to(self.device)
                    
                    if x.dim() == 3:
                        x = x.unsqueeze(1)
                    
                    outputs, auxiliary = self.model(x, adjacency)
                    
                    outputs_aligned, targets_aligned = self._align_shapes_for_metrics(outputs, y, -1)
                    loss = self.criterion(outputs_aligned, targets_aligned)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                     float(self.config['training'].get('gradient_clip', 1.0)))
                        target_optimizer.step()
                        epoch_loss += loss.item()
                        num_updates += 1
                    
                    if batch_idx >= 10:
                        break
                
                avg_loss = epoch_loss / max(num_updates, 1)
                
                if epoch % 10 == 0 or epoch == target_epochs - 1:
                    val_metrics = self.evaluate_on_target(target_loader, 3)
                    mae = val_metrics.get('MAE', 0.0)
                    rmse = val_metrics.get('RMSE', 0.0)
                    
                    print(f"Target Epoch {epoch:3d}/{target_epochs} | "
                          f"Loss: {avg_loss:.6f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
                else:
                    print(f"Target Epoch {epoch:3d}/{target_epochs} | Loss: {avg_loss:.6f}")
        
        total_time = self.timer.stop()
        elapsed_str = format_time(total_time)
        
        self.logger.log(f"Two-phase training completed in {elapsed_str}")
        print(f"\n" + "="*60)
        print(f"TRAINING COMPLETED: Source + Target Domain Adaptation")
        print(f"Total Training Time: {elapsed_str}")
        print("="*60)
        
        return training_history


class StandardTrainer:
    def __init__(self, model: ChaosAwareTrafficPredictor, config: Dict, device: torch.device, logger):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        
        self.criterion = nn.MSELoss()
        
        self.data_manager = ChaosDataManager(config)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['training'].get('learning_rate', 0.0005)),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            mode=config['training']['scheduler']['type'],
            factor=float(config['training']['scheduler']['factor']),
            patience=int(config['training']['scheduler']['patience']),
            min_lr=float(config['training']['scheduler']['min_lr'])
        )
        
        self.metrics_calculator = MetricsCalculator()
        
        self.early_stopping = EarlyStopping(
            patience=int(config.get('training', {}).get('early_stopping', {}).get('patience', 15)),
            min_delta=float(config.get('training', {}).get('early_stopping', {}).get('min_delta', 1e-5)),
            restore_best=True
        )
        self.timer = Timer()
    
    def _align_shapes_for_metrics(self, outputs, targets, batch_idx=0):
        outputs_aligned = outputs
        targets_aligned = targets
        
        if outputs.shape != targets.shape:
            if outputs.dim() == 3 and targets.dim() == 2:
                targets_aligned = targets.unsqueeze(1).expand(-1, outputs.size(1), -1)
            elif outputs.dim() == 3 and targets.dim() == 3:
                if outputs.shape[1] == 1 and targets.shape[2] == 1:
                    outputs_aligned = outputs.squeeze(1)
                    targets_aligned = targets.squeeze(-1)
                elif outputs.shape[1] != targets.shape[1] and outputs.shape[2] == targets.shape[1]:
                    outputs_aligned = outputs.transpose(1, 2)
                elif outputs.shape[1] == targets.shape[2] and outputs.shape[2] == targets.shape[1]:
                    targets_aligned = targets.transpose(1, 2)
            elif outputs.dim() == 2 and targets.dim() == 3:
                targets_aligned = targets[:, :, 0] if targets.shape[2] > 0 else targets.squeeze(-1)
            elif outputs.dim() == 2 and targets.dim() == 2:
                if outputs.shape[1] != targets.shape[1]:
                    min_features = min(outputs.shape[1], targets.shape[1])
                    outputs_aligned = outputs[:, :min_features]
                    targets_aligned = targets[:, :min_features]
            else:
                batch_size = outputs.size(0)
                outputs_flat = outputs.view(batch_size, -1)
                targets_flat = targets.view(batch_size, -1)
                min_features = min(outputs_flat.size(1), targets_flat.size(1))
                outputs_aligned = outputs_flat[:, :min_features]
                targets_aligned = targets_flat[:, :min_features]
        
        return outputs_aligned, targets_aligned
    
    def train_step(self, batch_data):
        self.model.train()
        self.optimizer.zero_grad()
        
        x = batch_data['temporal_data'].to(self.device)
        y = batch_data['targets'].to(self.device)
        adjacency = batch_data['adjacency'].to(self.device)
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        outputs, auxiliary = self.model(x, adjacency)
        
        outputs_aligned, targets_aligned = self._align_shapes_for_metrics(outputs, y)
        loss = self.criterion(outputs_aligned, targets_aligned)
        
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         float(self.config['training'].get('gradient_clip', 1.0)))
            self.optimizer.step()
        
        return {'total_loss': loss.item()}
    
    def evaluate(self, dataloader):
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_losses = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                x = batch_data['temporal_data'].to(self.device)
                y = batch_data['targets'].to(self.device)
                adjacency = batch_data['adjacency'].to(self.device)
                
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                
                outputs, auxiliary = self.model(x, adjacency)
                
                outputs_aligned, targets_aligned = self._align_shapes_for_metrics(outputs, y, -1)
                loss = self.criterion(outputs_aligned, targets_aligned)
                
                if outputs_aligned.shape == targets_aligned.shape:
                    self.metrics_calculator.update(outputs_aligned, targets_aligned, loss.item())
                
                total_losses.append(loss.item())
                
                if batch_idx >= 20:
                    break
        
        metrics = self.metrics_calculator.compute_metrics()
        metrics['total_loss'] = np.mean(total_losses) if total_losses else 0.0
        
        return metrics
    
    def train(self):
        self.timer.start()
        self.logger.log("Starting standard training with ChaosDataManager")
        
        self.logger.log("Creating datasets with ChaosDataManager...")
        self.data_manager.create_datasets(add_physics_features=self.config['training']['add_physics_features'])
        
        dataloaders = self.data_manager.create_dataloaders(
            add_physics_features=self.config['training']['add_physics_features']
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        dataset_info = self.data_manager.get_dataset_info()
        self.logger.log("Dataset Information:")
        for split, info in dataset_info.items():
            self.logger.log(f"  {split}: {info['num_samples']} samples, "
                          f"{info['num_nodes']} nodes, {info['num_features']} features, "
                          f"Physics features: {info['physics_features_enabled']}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        epochs = min(int(self.config['training']['epochs']), 300)
        
        self.logger.log(f"Training for {epochs} epochs")
        print("="*60)
        print(f"STANDARD TRAINING ({epochs} epochs)")
        print(f"Dataset: {self.config['training']['test_dataset']}")
        print(f"Physics features: {self.config['training']['add_physics_features']}")
        print("="*60)
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch_data in enumerate(train_loader):
                step_losses = self.train_step(batch_data)
                epoch_losses.append(step_losses['total_loss'])
                
                if batch_idx >= 20:
                    break
            
            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            training_history['train_loss'].append(avg_train_loss)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['total_loss']
                
                training_history['val_loss'].append(val_loss)
                training_history['val_metrics'].append(val_metrics)
                
                mae = val_metrics.get('MAE', 0.0)
                rmse = val_metrics.get('RMSE', 0.0)
                
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")
                
                self.logger.log_metrics(epoch, avg_train_loss, val_loss, val_metrics)
                self.scheduler.step(val_loss)
                
                if self.early_stopping(val_loss, self.model):
                    elapsed_time = format_time(self.timer.elapsed())
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Training time: {elapsed_time}")
                    self.logger.log(f"Early stopping at epoch {epoch}. Time: {elapsed_time}")
                    break
            else:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.6f}")
        
        total_time = self.timer.stop()
        elapsed_str = format_time(total_time)
        
        self.logger.log(f"Training completed in {elapsed_str}")
        print(f"\n" + "="*60)
        print(f"TRAINING COMPLETED")
        print(f"Total Training Time: {elapsed_str}")
        print("="*60)
        
        return training_history


def create_trainer(model: ChaosAwareTrafficPredictor, config: Dict, device: torch.device, logger):
    if config.get('few_shot', {}).get('enabled', False):
        return DomainAdaptationTrainer(model, config, device, logger)
    else:
        return StandardTrainer(model, config, device, logger)