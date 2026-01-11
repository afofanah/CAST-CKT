import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import warnings
from typing import Dict, Tuple
from scipy import stats
import os
import pickle

warnings.filterwarnings('ignore')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_normalized_adj(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1), dtype=np.float32).reshape((-1,))
    D[D <= 1e-5] = 1e-5
    diag = np.reciprocal(np.sqrt(D)).astype(np.float32)
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                        diag.reshape((1, -1)))
    return A_wave.astype(np.float32)

def robust_normalize_adjacency(A):
    A = A.astype(np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.array(np.sum(A, axis=1), dtype=np.float32).reshape((-1,))
    D = np.maximum(D, 1e-5)
    diag = np.reciprocal(np.sqrt(D)).astype(np.float32)
    A_normalized = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_normalized.astype(np.float32)

def enhanced_physics_features(data, adjacency_matrix, dataset_name, data_args):
    num_timesteps, num_nodes, original_features = data.shape
    
    if adjacency_matrix.shape[0] != num_nodes:
        min_nodes = min(adjacency_matrix.shape[0], num_nodes)
        adjacency_matrix = adjacency_matrix[:min_nodes, :min_nodes]
        data = data[:, :min_nodes, :]
        num_nodes = min_nodes
    
    physics_features = np.zeros((num_timesteps, num_nodes, 4), dtype=np.float32)
    
    degree = np.sum(adjacency_matrix, axis=1)
    degree_normalized = degree / (np.max(degree) + 1e-8)
    
    if original_features > 0:
        flow_data = data[:, :, 0]
        
        physics_features[:, :, 0] = np.broadcast_to(
            degree_normalized[np.newaxis, :], 
            (num_timesteps, num_nodes)
        )
        
        if num_timesteps > 2:
            flow_variance = np.var(flow_data, axis=0)
            flow_variance_norm = flow_variance / (np.max(flow_variance) + 1e-8)
            physics_features[:, :, 1] = np.broadcast_to(
                flow_variance_norm[np.newaxis, :], 
                (num_timesteps, num_nodes)
            )
            
            temporal_gradient = np.gradient(flow_data, axis=0)
            temporal_std = np.std(temporal_gradient) + 1e-6
            temporal_gradient_norm = np.clip(temporal_gradient / temporal_std, -3, 3)
            physics_features[:, :, 3] = temporal_gradient_norm
        else:
            physics_features[:, :, 1] = 0.1
            physics_features[:, :, 3] = 0.0
        
        neighbor_influence = np.zeros((num_timesteps, num_nodes))
        for node in range(num_nodes):
            neighbors = np.where(adjacency_matrix[node, :] > 0)[0]
            if len(neighbors) > 0:
                neighbor_flows = flow_data[:, neighbors]
                neighbor_influence[:, node] = np.mean(neighbor_flows, axis=1)
            else:
                neighbor_influence[:, node] = flow_data[:, node]
        
        if isinstance(data_args, dict) and dataset_name in data_args:
            speed_mean = data_args[dataset_name].get('speed_mean', 0.0)
            speed_std = data_args[dataset_name].get('speed_std', 1.0)
            neighbor_influence_norm = (neighbor_influence - speed_mean) / speed_std
        else:
            neighbor_mean = np.mean(neighbor_influence)
            neighbor_std = np.std(neighbor_influence) + 1e-6
            neighbor_influence_norm = (neighbor_influence - neighbor_mean) / neighbor_std
        
        physics_features[:, :, 2] = np.clip(neighbor_influence_norm, -3, 3)
    else:
        physics_features[:, :, 0] = np.broadcast_to(
            degree_normalized[np.newaxis, :], 
            (num_timesteps, num_nodes)
        )
        physics_features[:, :, 1:] = 0.0
    
    combined_features = np.concatenate([data, physics_features], axis=2)
    return combined_features.astype(np.float32)

def create_spatiotemporal_targets(flow_data, adjacency_matrix):
    num_nodes, time_len = flow_data.shape
    spatial_target = np.zeros((num_nodes, time_len, 2), dtype=np.float32)
    temporal_target = np.zeros((num_nodes, time_len), dtype=np.float32)
    
    eigenvals, eigenvecs = np.linalg.eigh(adjacency_matrix + 1e-6 * np.eye(num_nodes))
    
    for node in range(num_nodes):
        neighbors = np.where(adjacency_matrix[node, :] > 0)[0]
        
        node_flow = flow_data[node, :]
        
        if eigenvecs.shape[1] >= 2:
            spatial_embedding = eigenvecs[node, :2]
        else:
            spatial_embedding = np.array([node % 10, node // 10], dtype=np.float32)
        
        for t in range(time_len):
            flow_magnitude = np.abs(node_flow[t]) + 1e-6
            time_factor = 2 * np.pi * t / max(time_len, 1)
            
            spatial_target[node, t, 0] = spatial_embedding[0] + 0.1 * flow_magnitude * np.cos(time_factor)
            spatial_target[node, t, 1] = spatial_embedding[1] + 0.1 * flow_magnitude * np.sin(time_factor)
        
        if len(neighbors) > 0 and time_len > 1:
            neighbor_flows = flow_data[neighbors, :]
            neighbor_mean = np.mean(neighbor_flows, axis=0)
            correlation = np.corrcoef(node_flow, neighbor_mean)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            temporal_target[node, :] = correlation * np.gradient(node_flow)
        else:
            temporal_target[node, :] = np.gradient(node_flow) if time_len > 1 else np.zeros(time_len)
    
    spatial_mean = np.mean(spatial_target, axis=(0, 1), keepdims=True)
    spatial_std = np.std(spatial_target, axis=(0, 1), keepdims=True) + 1e-6
    spatial_target = (spatial_target - spatial_mean) / spatial_std
    
    temporal_mean = np.mean(temporal_target)
    temporal_std = np.std(temporal_target) + 1e-6
    temporal_target = (temporal_target - temporal_mean) / temporal_std
    
    return spatial_target, temporal_target

def robust_traffic_normalization(data: np.ndarray, method: str = 'improved_standard', 
                                epsilon: float = 1e-6, clip_outliers: bool = True,
                                outlier_threshold: float = 2.5) -> Tuple[np.ndarray, Dict]:
    
    original_shape = data.shape
    if data.ndim == 3:
        data_flat = data.reshape(-1, data.shape[-1])
    else:
        data_flat = data.copy()
    
    normalized_data = np.zeros_like(data_flat, dtype=np.float32)
    stats_info = {}
    
    if clip_outliers:
        q1, q3 = np.percentile(data_flat, [25, 75], axis=0)
        iqr = q3 - q1
        lower_bounds = q1 - outlier_threshold * iqr
        upper_bounds = q3 + outlier_threshold * iqr
        data_flat = np.clip(data_flat, lower_bounds, upper_bounds)
    
    for feature_idx in range(data_flat.shape[-1]):
        feature_data = data_flat[:, feature_idx]
        
        if method == 'improved_standard':
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            
            if std < epsilon:
                std = 1.0
                print(f"Warning: Feature {feature_idx} has very low variance, using std=1.0")
            
            normalized_feature = (feature_data - mean) / std
            
        elif method == 'robust_standard':
            median = np.median(feature_data)
            mad = np.median(np.abs(feature_data - median))
            scale = max(mad * 1.4826, epsilon)
            normalized_feature = (feature_data - median) / scale
            
        elif method == 'minmax_centered':
            min_val = np.min(feature_data)
            max_val = np.max(feature_data)
            range_val = max_val - min_val
            
            if range_val < epsilon:
                range_val = 1.0
                print(f"Warning: Feature {feature_idx} has very low range, using range=1.0")
            
            normalized_feature = 2 * (feature_data - min_val) / range_val - 1
            
        elif method == 'quantile_normal':
            sorted_indices = np.argsort(feature_data)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(feature_data))
            uniform = (ranks + 0.5) / len(feature_data)
            normalized_feature = stats.norm.ppf(uniform)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized_feature = np.clip(normalized_feature, -10.0, 10.0)
        normalized_data[:, feature_idx] = normalized_feature.astype(np.float32)
        
        stats_info[f'feature_{feature_idx}'] = {
            'method': method,
            'original_mean': float(np.mean(feature_data)),
            'original_std': float(np.std(feature_data)),
            'normalized_mean': float(np.mean(normalized_feature)),
            'normalized_std': float(np.std(normalized_feature))
        }
    
    if len(original_shape) == 3:
        normalized_data = normalized_data.reshape(original_shape)
    
    return normalized_data, stats_info

def choose_normalization_method(data: np.ndarray) -> str:
    if data.ndim == 3:
        main_feature = data[:min(1000, data.shape[0]), :, 0].flatten()
    else:
        main_feature = data.flatten()[:min(10000, len(data.flatten()))]
    
    zeros_ratio = np.mean(main_feature == 0)
    
    if zeros_ratio > 0.15:
        return 'robust_standard'
    
    mean_val = np.mean(main_feature)
    std_val = np.std(main_feature)
    
    if abs(mean_val) > 3 * std_val:
        return 'improved_standard'
    
    q25, q50, q75 = np.percentile(main_feature, [25, 50, 75])
    if q50 - q25 != 0 and q75 - q50 != 0:
        skewness_approx = ((q75 - q50) - (q50 - q25)) / (q75 - q25)
        if abs(skewness_approx) > 0.7:
            return 'quantile_normal'
    
    iqr = q75 - q25
    if iqr > 0:
        outliers = np.sum((main_feature < q25 - 3*iqr) | (main_feature > q75 + 3*iqr))
        outlier_ratio = outliers / len(main_feature)
        if outlier_ratio > 0.1:
            return 'robust_standard'
    
    return 'improved_standard'

def load_real_traffic_data(dataset_path, adjacency_path, normalize=True, dataset_config=None, 
                          add_physics_features=True):
    
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = f"enhanced_{os.path.basename(dataset_path)}_{os.path.basename(adjacency_path)}_{normalize}_{add_physics_features}_v3"
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    traffic_data = np.load(dataset_path).astype(np.float32)
    adjacency_matrix = np.load(adjacency_path).astype(np.float32)
    
    if len(traffic_data.shape) == 3:
        if traffic_data.shape[0] > traffic_data.shape[2] and traffic_data.shape[1] < 1000:
            pass
        elif traffic_data.shape[2] > traffic_data.shape[0] and traffic_data.shape[1] < 1000:
            traffic_data = traffic_data.transpose((2, 0, 1))
        elif traffic_data.shape[1] > traffic_data.shape[0] and traffic_data.shape[1] > traffic_data.shape[2]:
            traffic_data = traffic_data.transpose((1, 2, 0))
        else:
            traffic_data = traffic_data.transpose((2, 0, 1))
    elif len(traffic_data.shape) == 2:
        num_timesteps, num_nodes = traffic_data.shape
        traffic_data = traffic_data[:, :, np.newaxis]
        num_features = 1
    else:
        raise ValueError(f"Expected 2D or 3D traffic data, got shape {traffic_data.shape}")
    
    num_timesteps, num_nodes, num_features = traffic_data.shape
    
    traffic_data = np.nan_to_num(
        traffic_data, 
        nan=0.0, 
        posinf=np.percentile(traffic_data[np.isfinite(traffic_data)], 99), 
        neginf=np.percentile(traffic_data[np.isfinite(traffic_data)], 1)
    )
    
    if dataset_config:
        expected_nodes = dataset_config.get('node_num')
        expected_timesteps = dataset_config.get('time_step')
        
        if expected_nodes and num_nodes != expected_nodes:
            print(f"Warning: Expected {expected_nodes} nodes, got {num_nodes}")
        if expected_timesteps and num_timesteps != expected_timesteps:
            print(f"Warning: Expected {expected_timesteps} timesteps, got {num_timesteps}")
    
    print(f"Original data stats - Mean: {np.mean(traffic_data):.6f}, Std: {np.std(traffic_data):.6f}")
    print(f"Original data range: [{np.min(traffic_data):.6f}, {np.max(traffic_data):.6f}]")
    
    if normalize:
        method = choose_normalization_method(traffic_data)
        normalized_data, norm_stats = robust_traffic_normalization(
            traffic_data, 
            method=method,
            clip_outliers=True,
            outlier_threshold=2.5
        )
        traffic_data = normalized_data
        print(f"Applied {method} normalization to traffic data")
        print(f"Normalized data stats - Mean: {np.mean(traffic_data):.6f}, Std: {np.std(traffic_data):.6f}")
        print(f"Normalized data range: [{np.min(traffic_data):.6f}, {np.max(traffic_data):.6f}]")
    
    if add_physics_features:
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        data_args = {dataset_name: {'speed_mean': np.mean(traffic_data), 'speed_std': np.std(traffic_data)}}
        
        traffic_data = enhanced_physics_features(
            traffic_data, adjacency_matrix, dataset_name, data_args
        )
        
        print(f"Added physics features. New shape: {traffic_data.shape}")
        num_features = traffic_data.shape[-1]
    
    result = {
        'data': traffic_data,
        'adjacency_matrix': adjacency_matrix,
        'num_nodes': num_nodes,
        'num_timesteps': num_timesteps,
        'num_features': num_features
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    
    return result

def enhanced_sequences(X, sequence_length, prediction_horizons, adjacency_matrix=None):
    if not isinstance(prediction_horizons, list):
        prediction_horizons = [prediction_horizons]
    
    max_horizon = max(prediction_horizons)
    total_len = sequence_length + max_horizon
    num_sequences = max(1, X.shape[0] - total_len + 1)
    
    step_size = max(1, num_sequences // 800) if num_sequences > 800 else 1
    indices = list(range(0, num_sequences, step_size))
    
    sequences = []
    targets = []
    
    for i in indices:
        sequence_data = X[i:i + sequence_length]
        
        horizon_targets = []
        for horizon in prediction_horizons:
            target_idx = i + sequence_length
            target = X[target_idx:target_idx + horizon]
            horizon_targets.append(target[:, :, 0])
        
        target_combined = np.concatenate(horizon_targets, axis=0) if len(horizon_targets) > 1 else horizon_targets[0]
        
        if np.any(np.isnan(sequence_data)) or np.any(np.isnan(target_combined)):
            continue
            
        sequences.append(sequence_data)
        targets.append(target_combined)
    
    if not sequences:
        dummy_sequence = np.zeros((1, sequence_length, X.shape[1], X.shape[2]), dtype=np.float32)
        dummy_target = np.zeros((1, X.shape[1], max_horizon), dtype=np.float32)
        return dummy_sequence, dummy_target
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

class MultiHorizonTrafficDataset(Dataset):
    def __init__(self, traffic_data, adjacency_matrix, sequence_length=6, 
                 prediction_horizons=[1], chaos_dim=16, add_physics_features=True):
        self.traffic_data = traffic_data
        self.adjacency_matrix = adjacency_matrix
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons if isinstance(prediction_horizons, list) else [prediction_horizons]
        self.max_horizon = max(self.prediction_horizons)
        self.chaos_dim = chaos_dim
        self.add_physics_features = add_physics_features
        
        self.num_timesteps, self.num_nodes, self.num_features = traffic_data.shape
        self.num_sequences = self.num_timesteps - sequence_length - self.max_horizon + 1
        
        if self.num_sequences <= 0:
            raise ValueError(f"Not enough data for sequence_length={sequence_length} and max_horizon={self.max_horizon}")
        
        self.edge_index, self.edge_attr = self._create_edge_data_optimized()
        self._precompute_chaos_features()
        
    def _create_edge_data_optimized(self):
        edge_indices = np.nonzero(self.adjacency_matrix)
        
        edge_list = np.column_stack((edge_indices[0], edge_indices[1]))
        edge_weights = self.adjacency_matrix[edge_indices]
        
        edge_index = torch.tensor(edge_list.T, dtype=torch.long).contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def _precompute_chaos_features(self):
        self.chaos_features_cache = torch.zeros(self.num_sequences, self.chaos_dim, dtype=torch.float32)
        
        for idx in range(self.num_sequences):
            sequence_data = self.traffic_data[idx:idx + self.sequence_length]
            flow_sequence = sequence_data[:, :, 0]
            
            features = np.zeros(self.chaos_dim)
            if flow_sequence.size > 0:
                flat_sequence = flow_sequence.flatten()
                
                features[0] = np.mean(flat_sequence)
                features[1] = np.std(flat_sequence)
                features[2] = np.var(flat_sequence)
                features[3] = np.max(flat_sequence) - np.min(flat_sequence)
                
                if self.chaos_dim > 4:
                    features[4] = np.percentile(flat_sequence, 75) - np.percentile(flat_sequence, 25)
                if self.chaos_dim > 5:
                    temporal_diffs = np.diff(flat_sequence)
                    features[5] = np.std(temporal_diffs) if len(temporal_diffs) > 0 else 0
                if self.chaos_dim > 6:
                    features[6] = np.mean(np.abs(temporal_diffs)) if len(temporal_diffs) > 0 else 0
                if self.chaos_dim > 7:
                    spatial_var = np.var(flow_sequence, axis=1)
                    features[7] = np.mean(spatial_var)
                
                if self.chaos_dim > 8:
                    autocorr_lag1 = np.corrcoef(flat_sequence[:-1], flat_sequence[1:])[0, 1] if len(flat_sequence) > 1 else 0
                    features[8] = 0 if np.isnan(autocorr_lag1) else autocorr_lag1
                    
                if self.chaos_dim > 9:
                    features[9] = len(np.where(np.diff(flat_sequence) > 0)[0]) / len(flat_sequence) if len(flat_sequence) > 1 else 0.5
                    
                if self.chaos_dim > 10:
                    q1, q3 = np.percentile(flat_sequence, [25, 75])
                    iqr = q3 - q1
                    features[10] = iqr / (np.std(flat_sequence) + 1e-8)
                    
                if self.chaos_dim > 11:
                    flow_entropy = -np.sum(np.histogram(flat_sequence, bins=10)[0] / len(flat_sequence) * 
                                         np.log(np.histogram(flat_sequence, bins=10)[0] / len(flat_sequence) + 1e-8))
                    features[11] = flow_entropy
                    
                if self.chaos_dim > 12:
                    features[12] = np.mean(np.abs(sequence_data[:, :, 1])) if sequence_data.shape[-1] > 1 else 0
                    
                if self.chaos_dim > 13:
                    features[13] = np.mean(sequence_data[:, :, -1]) if sequence_data.shape[-1] > 1 else 0
                    
                if self.chaos_dim > 14:
                    spectral_centroid = np.mean(np.abs(np.fft.fft(flat_sequence - np.mean(flat_sequence))))
                    features[14] = spectral_centroid
                    
                if self.chaos_dim > 15:
                    persistence = len(np.where(np.diff(np.sign(np.diff(flat_sequence))) != 0)[0]) / len(flat_sequence) if len(flat_sequence) > 2 else 0
                    features[15] = persistence
            
            features = np.clip(features, -50, 50)
            norm = np.linalg.norm(features) + 1e-8
            features = features / norm * 2.0
            
            self.chaos_features_cache[idx, :] = torch.tensor(features, dtype=torch.float32)
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        input_sequence = self.traffic_data[idx:idx + self.sequence_length]
        
        targets = []
        for horizon in self.prediction_horizons:
            target_idx = idx + self.sequence_length
            target = self.traffic_data[target_idx:target_idx + horizon]
            targets.append(target[:, :, 0])
        
        target_combined = np.concatenate(targets, axis=0) if len(targets) > 1 else targets[0]
        
        return {
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'temporal_data': torch.FloatTensor(input_sequence),
            'targets': torch.FloatTensor(target_combined),
            'adjacency': torch.FloatTensor(self.adjacency_matrix),
            'chaos_features': self.chaos_features_cache[idx]
        }

class FastRealTrafficDataset(Dataset):
    def __init__(self, traffic_data, adjacency_matrix, sequence_length=6, 
                 prediction_horizon=1, chaos_dim=16, add_physics_features=True):
        self.traffic_data = traffic_data
        self.adjacency_matrix = adjacency_matrix
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.chaos_dim = chaos_dim
        self.add_physics_features = add_physics_features
        
        self.num_timesteps, self.num_nodes, self.num_features = traffic_data.shape
        self.num_sequences = self.num_timesteps - sequence_length - prediction_horizon + 1
        
        self.edge_index, self.edge_attr = self._create_edge_data_optimized()
        self._precompute_chaos_features()
        
    def _create_edge_data_optimized(self):
        adjacency_density = np.count_nonzero(self.adjacency_matrix) / (self.adjacency_matrix.shape[0] ** 2)
        
        if adjacency_density > 0.5:
            print(f"Dense adjacency matrix detected (density: {adjacency_density:.3f}), skipping edge data creation")
            return None, None
        
        edge_indices = np.nonzero(self.adjacency_matrix)
        
        if len(edge_indices[0]) > 50000:
            print(f"Too many edges ({len(edge_indices[0])}), limiting to top 50000")
            edge_weights = self.adjacency_matrix[edge_indices]
            top_indices = np.argsort(edge_weights)[-50000:]
            edge_indices = (edge_indices[0][top_indices], edge_indices[1][top_indices])
        
        edge_list = np.column_stack((edge_indices[0], edge_indices[1]))
        edge_weights = self.adjacency_matrix[edge_indices]
        
        edge_index = torch.tensor(edge_list.T, dtype=torch.long).contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        
        print(f"Created edge data: {edge_index.shape[1]} edges for {self.adjacency_matrix.shape[0]} nodes")
        
        return edge_index, edge_attr
    
    def _precompute_chaos_features(self):
        self.chaos_features_cache = torch.zeros(self.num_sequences, self.chaos_dim, dtype=torch.float32)
        
        for idx in range(self.num_sequences):
            sequence_data = self.traffic_data[idx:idx + self.sequence_length]
            flow_sequence = sequence_data[:, :, 0]
            
            features = np.zeros(self.chaos_dim)
            if flow_sequence.size > 0:
                flat_sequence = flow_sequence.flatten()
                
                features[0] = np.mean(flat_sequence)
                features[1] = np.std(flat_sequence)
                features[2] = np.var(flat_sequence)
                features[3] = np.max(flat_sequence) - np.min(flat_sequence)
                
                if self.chaos_dim > 4:
                    features[4] = np.percentile(flat_sequence, 75) - np.percentile(flat_sequence, 25)
                if self.chaos_dim > 5:
                    temporal_diffs = np.diff(flat_sequence)
                    features[5] = np.std(temporal_diffs) if len(temporal_diffs) > 0 else 0
                if self.chaos_dim > 6:
                    features[6] = np.mean(np.abs(temporal_diffs)) if len(temporal_diffs) > 0 else 0
                if self.chaos_dim > 7:
                    spatial_var = np.var(flow_sequence, axis=1)
                    features[7] = np.mean(spatial_var)
                
                if self.chaos_dim > 8:
                    autocorr_lag1 = np.corrcoef(flat_sequence[:-1], flat_sequence[1:])[0, 1] if len(flat_sequence) > 1 else 0
                    features[8] = 0 if np.isnan(autocorr_lag1) else autocorr_lag1
                    
                if self.chaos_dim > 9:
                    features[9] = len(np.where(np.diff(flat_sequence) > 0)[0]) / len(flat_sequence) if len(flat_sequence) > 1 else 0.5
                    
                if self.chaos_dim > 10:
                    q1, q3 = np.percentile(flat_sequence, [25, 75])
                    iqr = q3 - q1
                    features[10] = iqr / (np.std(flat_sequence) + 1e-8)
                    
                if self.chaos_dim > 11:
                    flow_entropy = -np.sum(np.histogram(flat_sequence, bins=10)[0] / len(flat_sequence) * 
                                         np.log(np.histogram(flat_sequence, bins=10)[0] / len(flat_sequence) + 1e-8))
                    features[11] = flow_entropy
                    
                if self.chaos_dim > 12:
                    features[12] = np.mean(np.abs(sequence_data[:, :, 1])) if sequence_data.shape[-1] > 1 else 0
                    
                if self.chaos_dim > 13:
                    features[13] = np.mean(sequence_data[:, :, -1]) if sequence_data.shape[-1] > 1 else 0
                    
                if self.chaos_dim > 14:
                    spectral_centroid = np.mean(np.abs(np.fft.fft(flat_sequence - np.mean(flat_sequence))))
                    features[14] = spectral_centroid
                    
                if self.chaos_dim > 15:
                    persistence = len(np.where(np.diff(np.sign(np.diff(flat_sequence))) != 0)[0]) / len(flat_sequence) if len(flat_sequence) > 2 else 0
                    features[15] = persistence
            
            features = np.clip(features, -50, 50)
            norm = np.linalg.norm(features) + 1e-8
            features = features / norm * 2.0
            
            self.chaos_features_cache[idx, :] = torch.tensor(features, dtype=torch.float32)
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        input_sequence = self.traffic_data[idx:idx + self.sequence_length]
        target_idx = idx + self.sequence_length
        target = self.traffic_data[target_idx:target_idx + self.prediction_horizon]
        
        target_flow = target[:, :, 0]
        
        return {
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'temporal_data': torch.FloatTensor(input_sequence),
            'targets': torch.FloatTensor(target_flow),
            'adjacency': torch.FloatTensor(self.adjacency_matrix),
            'chaos_features': self.chaos_features_cache[idx]
        }

class ChaosDataManager:
    def __init__(self, config: Dict):
        self.config = config
        self.datasets = {}
        self._data_cache = {}
        
    def create_datasets(self, data_path: str = None, adjacency_path: str = None, 
                       add_physics_features: bool = True):
        if data_path is None or adjacency_path is None:
            dataset_name = self.config['training'].get('test_dataset', 'metr-la')
            dataset_config = self.config['data'][dataset_name]
            data_path = dataset_config['dataset_path']
            adjacency_path = dataset_config['adjacency_matrix_path']
        else:
            dataset_name = self.config['training'].get('test_dataset', 'metr-la')
            dataset_config = self.config['data'].get(dataset_name, {})
        
        cache_key = f"{data_path}_{adjacency_path}_{add_physics_features}"
        if cache_key in self._data_cache:
            data_info = self._data_cache[cache_key]
        else:
            preprocessing_config = self.config.get('preprocessing', {})
            normalize = True
            
            data_info = load_real_traffic_data(
                data_path,
                adjacency_path,
                normalize=normalize,
                dataset_config=dataset_config,
                add_physics_features=add_physics_features
            )
            self._data_cache[cache_key] = data_info
        
        traffic_data = data_info['data']
        adjacency_matrix = data_info['adjacency_matrix']
        
        train_ratio = 0.7
        val_ratio = 0.1
        
        num_timesteps = traffic_data.shape[0]
        train_size = int(train_ratio * num_timesteps)
        val_size = int(val_ratio * num_timesteps)
        
        train_data = traffic_data[:train_size]
        val_data = traffic_data[train_size:train_size + val_size]
        test_data = traffic_data[train_size + val_size:]
        
        sequence_length = self.config['task']['his_num']
        prediction_horizon = self.config['task']['pred_num']
        chaos_dim = self.config['model']['chaos_dim']
        
        print(f"Creating datasets with seq_len={sequence_length}, pred_horizon={prediction_horizon}")
        print(f"Physics features enabled: {add_physics_features}")
        print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        
        self.datasets['train'] = FastRealTrafficDataset(
            train_data, adjacency_matrix, sequence_length, prediction_horizon, chaos_dim, add_physics_features
        )
        
        self.datasets['val'] = FastRealTrafficDataset(
            val_data, adjacency_matrix, sequence_length, prediction_horizon, chaos_dim, add_physics_features
        )
        
        self.datasets['test'] = FastRealTrafficDataset(
            test_data, adjacency_matrix, sequence_length, prediction_horizon, chaos_dim, add_physics_features
        )
    
    def create_dataloaders(self, add_physics_features: bool = True):
        dataloaders = {}
        
        num_workers = 0
        pin_memory = torch.cuda.is_available()
        prefetch_factor = None if num_workers == 0 else 2
        
        dataloaders['train'] = DataLoader(
            self.datasets['train'],
            batch_size=self.config['task']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            prefetch_factor=prefetch_factor
        )
        
        dataloaders['val'] = DataLoader(
            self.datasets['val'],
            batch_size=self.config['task']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            prefetch_factor=prefetch_factor
        )
        
        dataloaders['test'] = DataLoader(
            self.datasets['test'],
            batch_size=self.config['task'].get('test_batch_size', self.config['task']['batch_size']),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            prefetch_factor=prefetch_factor
        )
        
        return dataloaders
    
    def get_dataset_info(self):
        info = {}
        for split, dataset in self.datasets.items():
            info[split] = {
                'num_samples': len(dataset),
                'num_nodes': dataset.num_nodes,
                'sequence_length': dataset.sequence_length,
                'prediction_length': dataset.prediction_horizon if hasattr(dataset, 'prediction_horizon') else dataset.max_horizon,
                'num_features': dataset.num_features,
                'physics_features_enabled': dataset.add_physics_features
            }
        
        return info
    
    def get_sample_data(self):
        """Get a sample of the data to determine dimensions"""
        if not hasattr(self, 'datasets') or 'train' not in self.datasets:
            return {}
        
        if len(self.datasets['train']) > 0:
            sample = self.datasets['train'][0]
            return sample
        
        return {}