import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from scipy.spatial.distance import pdist, squareform
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.utils.checkpoint as checkpoint
from functools import lru_cache

class ChaosAnalyzer:
    def __init__(self, embedding_dim=3, time_delay=1, max_embed_dim=10):
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.max_embed_dim = max_embed_dim
        self.feature_cache = {}
        
    def phase_space_reconstruction(self, time_series):
        n = len(time_series)
        if n < self.embedding_dim * self.time_delay:
            return np.zeros((1, self.embedding_dim))
        
        num_points = n - (self.embedding_dim - 1) * self.time_delay
        embedded = np.zeros((num_points, self.embedding_dim))

        indices = np.arange(num_points)[:, None] + np.arange(self.embedding_dim)[None, :] * self.time_delay
        embedded = time_series[indices]
        
        return embedded
    
    def largest_lyapunov_exponent(self, time_series):
        embedded = self.phase_space_reconstruction(time_series)
        if embedded.shape[0] < 10:
            return 0.0
            
        N = embedded.shape[0]
        
        diff = embedded[:, None, :] - embedded[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        nearest_neighbors = []
        for i in range(N - 20):
            mask = np.ones(N, dtype=bool)
            mask[max(0, i-1):min(N, i+2)] = False
            
            if np.any(mask):
                valid_distances = distances[i, mask]
                valid_indices = np.where(mask)[0]
                min_idx = valid_indices[np.argmin(valid_distances)]
                nearest_neighbors.append((i, min_idx, distances[i, min_idx]))
        
        if not nearest_neighbors:
            return 0.0
        
        lyapunov_sum = 0.0
        count = 0
        
        for i, j, _ in nearest_neighbors[:min(len(nearest_neighbors), 20)]:
            divergences = []
            max_k = min(20, N - max(i, j))
            
            if max_k > 1:
                k_range = np.arange(1, max_k)
                i_indices = i + k_range
                j_indices = j + k_range
                
                valid_mask = (i_indices < N) & (j_indices < N)
                if np.any(valid_mask):
                    i_valid = i_indices[valid_mask]
                    j_valid = j_indices[valid_mask]
                    
                    dists = np.linalg.norm(embedded[i_valid] - embedded[j_valid], axis=1)
                    valid_dists = dists[dists > 0]
                    
                    if len(valid_dists) > 5:
                        divergences = np.log(valid_dists)
                        times = np.arange(1, len(divergences) + 1)
                        slope = np.polyfit(times, divergences, 1)[0]
                        lyapunov_sum += slope
                        count += 1
        
        return lyapunov_sum / max(count, 1)
    
    def hurst_exponent(self, time_series):
        n = len(time_series)
        if n < 20:
            return 0.5
        
        sizes = np.unique(np.logspace(1, np.log10(n//2), 10).astype(int))
        rs_values = []
        
        for size in sizes:
            if size >= n:
                continue
   
            num_windows = n // size
            rs_window = []
            
            for i in range(num_windows):
                start_idx = i * size
                end_idx = start_idx + size
                window = time_series[start_idx:end_idx]
                
                if len(window) == size:
                    mean_window = np.mean(window)
                    detrended = window - mean_window
                    cumsum = np.cumsum(detrended)
                    
                    R = np.ptp(cumsum)  
                    S = np.std(window)
                    
                    if S > 0:
                        rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) < 3:
            return 0.5
        
        log_sizes = np.log(sizes[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        hurst = np.polyfit(log_sizes, log_rs, 1)[0]
        return max(0.0, min(hurst, 1.0))
    
    def sample_entropy(self, time_series, m=2):
        N = len(time_series)
        if N < 10:
            return 0.0
        
        r = 0.2 * np.std(time_series)
        
        def _phi_vectorized(m):
            patterns = np.array([time_series[i:i + m] for i in range(N - m + 1)])
            phi_sum = 0.0
            for i in range(N - m + 1):
                template = patterns[i]
                diffs = np.abs(patterns - template[None, :])
                max_diffs = np.max(diffs, axis=1)
                matches = np.sum(max_diffs <= r)
                
                if matches > 0:
                    phi_sum += np.log(matches / float(N - m + 1))
            
            return phi_sum / float(N - m + 1)
        
        return _phi_vectorized(m) - _phi_vectorized(m + 1)
    
    def correlation_dimension(self, time_series, max_r=None):
        embedded = self.phase_space_reconstruction(time_series)
        N = embedded.shape[0]
        
        if N < 20:
            return 1.0
        
        if max_r is None:
            distances = np.sqrt(np.sum((embedded[:, None, :] - embedded[None, :, :])**2, axis=2))
            distances_flat = distances[np.triu_indices(N, k=1)]
            max_r = np.percentile(distances_flat, 90)
        
        r_values = np.logspace(np.log10(max_r/100), np.log10(max_r), 20)
        correlation_sums = []
        distances = np.sqrt(np.sum((embedded[:, None, :] - embedded[None, :, :])**2, axis=2))
        
        for r in r_values:
            count = np.sum(distances < r) - N  
            correlation_sum = count / (N * (N - 1))
            correlation_sums.append(max(correlation_sum, 1e-10))
        
        log_r = np.log(r_values)
        log_c = np.log(correlation_sums)
        
        start_idx = int(0.2 * len(log_r))
        end_idx = int(0.8 * len(log_r))
        
        if end_idx > start_idx + 2:
            slope = np.polyfit(log_r[start_idx:end_idx], log_c[start_idx:end_idx], 1)[0]
            return max(0.1, min(slope, 10.0))
        
        return 1.0
    
    def box_counting_dimension(self, time_series):
        embedded = self.phase_space_reconstruction(time_series)
        if embedded.shape[0] < 10:
            return 1.0
        
        min_vals = np.min(embedded, axis=0)
        max_vals = np.max(embedded, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        normalized = (embedded - min_vals) / range_vals
        
        box_sizes = np.logspace(-2, 0, 10)
        box_counts = []
        
        for box_size in box_sizes:
            box_coords = np.floor(normalized / box_size).astype(int)
            unique_boxes = set(map(tuple, box_coords))
            box_counts.append(len(unique_boxes))
        
        log_sizes = np.log(1/box_sizes)
        log_counts = np.log(box_counts)
        
        if len(log_sizes) > 3:
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            return max(0.1, min(slope, 3.0))
        
        return 1.0
    
    def recurrence_rate(self, time_series, threshold_percent=10):
        embedded = self.phase_space_reconstruction(time_series)
        N = embedded.shape[0]
        
        if N < 10:
            return 0.0
        
        distances = np.sqrt(np.sum((embedded[:, None, :] - embedded[None, :, :])**2, axis=2))
        threshold = np.percentile(distances, threshold_percent)
        recurrence_matrix = distances < threshold
        recurrence_rate = np.sum(recurrence_matrix) / (N * N)
        
        return recurrence_rate
    
    def multifractal_spectrum(self, time_series):
        if len(time_series) < 32:
            return 0.0
        
        N = len(time_series)
        q_values = np.arange(-5, 6, 2)
        tau_q = []
        
        for q in q_values:
            scales = np.unique(np.logspace(1, np.log10(N//4), 8).astype(int))
            log_scales = []
            log_fluctuations = []
            
            for scale in scales:
                if scale >= N:
                    continue
                    
                num_boxes = N // scale
                fluctuations = []
                for i in range(num_boxes):
                    start_idx = i * scale
                    end_idx = start_idx + scale
                    segment = time_series[start_idx:end_idx]
                    
                    if len(segment) == scale:
                        x = np.arange(len(segment))
                        coeffs = np.polyfit(x, segment, 1)
                        trend = np.polyval(coeffs, x)
                        fluctuation = np.sqrt(np.mean((segment - trend)**2))
                        if fluctuation > 0:
                            fluctuations.append(fluctuation)
                
                if fluctuations:
                    fluctuations = np.array(fluctuations)
                    if q == 0:
                        avg_fluctuation = np.exp(np.mean(np.log(fluctuations)))
                    else:
                        avg_fluctuation = np.mean(fluctuations**q)**(1/q)
                    
                    log_scales.append(np.log(scale))
                    log_fluctuations.append(np.log(avg_fluctuation))
            
            if len(log_scales) > 2:
                tau_q.append(np.polyfit(log_scales, log_fluctuations, 1)[0])
            else:
                tau_q.append(0.0)
        
        if len(tau_q) > 1:
            return np.max(tau_q) - np.min(tau_q)
        return 0.0
    
    def approximate_entropy(self, time_series, m=2, r=None):
        N = len(time_series)
        if N < 10:
            return 0.0
        if r is None:
            r = 0.2 * np.std(time_series)
        
        def _phi_efficient(m):
            patterns = np.array([time_series[i:i + m] for i in range(N - m + 1)])
            phi_sum = 0.0
            for i in range(N - m + 1):
                template = patterns[i]
                diffs = np.abs(patterns - template)
                max_diffs = np.max(diffs, axis=1)
                matches = np.sum(max_diffs <= r)
                
                if matches > 0:
                    phi_sum += np.log(matches / float(N - m + 1))
            
            return phi_sum / float(N - m + 1) 
        return _phi_efficient(m) - _phi_efficient(m + 1)
    
    def extract_chaos_features(self, time_series):
        if len(time_series) < 10:
            return np.zeros(16)
        
        ts = np.array(time_series).flatten()
        ts = ts[np.isfinite(ts)]
        if len(ts) < 10:
            return np.zeros(16)
        cache_key = hash(tuple(ts[:100])) if len(ts) > 100 else hash(tuple(ts))
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = np.zeros(16)
        features[0] = self.largest_lyapunov_exponent(ts)
        features[1] = self.hurst_exponent(ts)
        features[2] = self.sample_entropy(ts)
        features[3] = self.correlation_dimension(ts)
        features[4] = self.box_counting_dimension(ts)
        features[5] = self.recurrence_rate(ts)
        features[6] = self.multifractal_spectrum(ts)
        features[7] = self.approximate_entropy(ts)
        
        features[8] = np.mean(ts)
        features[9] = np.std(ts)
        features[10] = np.var(ts)
        features[11] = len(ts)
        embedded = self.phase_space_reconstruction(ts)
        if embedded.shape[0] > 1:
            distances = pdist(embedded)
            features[12] = np.mean(distances)
            features[13] = np.std(distances)
            features[14] = np.max(distances)
            non_zero_distances = distances[distances > 0]
            features[15] = np.min(non_zero_distances) if len(non_zero_distances) > 0 else 0.0
        else:
            features[12:16] = 0.0
        result = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(self.feature_cache) < 100:  
            self.feature_cache[cache_key] = result
        
        return result


    
class ChaosAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, chaos_dim=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chaos_dim = chaos_dim
        self.qkv_linear = nn.Linear(embed_dim, embed_dim * 3)
        
        self.chaos_encoder = nn.Sequential(
            nn.Linear(chaos_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.chaos_gate_attention = nn.Sequential(
            nn.Linear(chaos_dim, num_heads),
            nn.Sigmoid()
        )
        
        self.chaos_gate_value = nn.Sequential(
            nn.Linear(chaos_dim, num_heads),
            nn.Sigmoid()
        )
        
        self.quantum_rotations = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.1)
        self.fractal_scales = nn.Parameter(torch.ones(num_heads) * 0.5)
        
        self.chaos_temperature = nn.Sequential(
            nn.Linear(chaos_dim, 1),
            nn.Softplus()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm_pre = nn.LayerNorm(embed_dim)
        self.norm_post = nn.LayerNorm(embed_dim)
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, x, chaos_features=None):
        original_shape = x.shape
        original_batch_size = x.size(0)
        
        x = self.norm_pre(x)
        if x.dim() == 4:
            batch_size, num_nodes, seq_len, embed_dim = x.size()
            x = x.view(batch_size * num_nodes, seq_len, embed_dim)
            effective_batch_size = batch_size * num_nodes
        elif x.dim() == 3:
            batch_size, seq_len, embed_dim = x.size()
            effective_batch_size = batch_size
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            batch_size, seq_len, embed_dim = x.size()
            effective_batch_size = batch_size
        else:
            raise ValueError(f"Unsupported input tensor shape: {x.shape}")

        qkv = self.qkv_linear(x).view(effective_batch_size, seq_len, 3, self.num_heads, self.head_dim)
        Q, K, V = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        if self.use_flash_attention and chaos_features is None:
            attended = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout.p if self.training else 0.0)
            attended = attended.transpose(1, 2).contiguous().view(effective_batch_size, seq_len, self.embed_dim)
        else:
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if chaos_features is not None:
                if chaos_features.dim() > 1:
                    chaos_features = chaos_features.view(-1)
                if chaos_features.size(0) != self.chaos_dim:
                    if chaos_features.size(0) > self.chaos_dim:
                        chaos_features = chaos_features[:self.chaos_dim]
                    else:
                        padding = torch.zeros(self.chaos_dim - chaos_features.size(0), device=chaos_features.device)
                        chaos_features = torch.cat([chaos_features, padding])
                
                chaos_encoded = self.chaos_encoder(chaos_features)
                chaos_encoded = chaos_encoded.view(self.num_heads, self.head_dim)
                
                chaos_gate_attn = self.chaos_gate_attention(chaos_features)
                chaos_gate_val = self.chaos_gate_value(chaos_features)
                temperature = self.chaos_temperature(chaos_features) + 1.0
                
                rotated_chaos = torch.matmul(chaos_encoded.unsqueeze(0), self.quantum_rotations)
                fractal_scales = torch.sigmoid(self.fractal_scales).unsqueeze(-1)
                scaled_chaos = rotated_chaos * fractal_scales.unsqueeze(-1)

                if scaled_chaos.numel() != self.num_heads * self.head_dim:
                    scaled_chaos = scaled_chaos.view(-1)[:self.num_heads * self.head_dim]
                    scaled_chaos = scaled_chaos.view(1, self.num_heads, self.head_dim)
                else:
                    scaled_chaos = scaled_chaos.view(1, self.num_heads, self.head_dim)
                chaos_bias = scaled_chaos.unsqueeze(2).expand(
                    effective_batch_size, self.num_heads, seq_len, self.head_dim
                )
                
                chaos_gate_expanded = chaos_gate_attn.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                    effective_batch_size, self.num_heads, seq_len, seq_len
                )
                chaos_attention_bias = torch.sum(chaos_bias, dim=-1)  
                chaos_attention_bias = chaos_attention_bias.unsqueeze(-1).expand(
                    effective_batch_size, self.num_heads, seq_len, seq_len
                )
                
                attn_scores = attn_scores * chaos_gate_expanded + chaos_attention_bias
                attn_scores = attn_scores / temperature
                
                chaos_value_bias = chaos_bias.mean(dim=2, keepdim=True).expand_as(V)
                chaos_gate_val_expanded = chaos_gate_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(V)
                V = V + chaos_value_bias * chaos_gate_val_expanded
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attended = torch.matmul(attn_weights, V)
            attended = attended.transpose(1, 2).contiguous().view(effective_batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attended)
        
        if chaos_features is not None:
            chaos_residual_gate = torch.sigmoid(torch.mean(chaos_features)) * 0.5 + 0.5
            output = output * chaos_residual_gate + x * (1 - chaos_residual_gate)
        else:
            output = output + x
        
        output = self.norm_post(output)
        if original_shape[0] != effective_batch_size:
            if len(original_shape) == 4:
                output = output.view(original_batch_size, original_shape[1], seq_len, self.embed_dim)
            else:
                output = output.view(original_shape[0], seq_len, self.embed_dim)
        
        return output, attn_weights if 'attn_weights' in locals() else None

class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_scales=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.short_term = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout)
        self.medium_term = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout)
        self.long_term = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        lstm_concat_dim = 3 * (hidden_dim // 8)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, lstm_concat_dim) * 0.1)
        self.lstm_projection = nn.Linear(lstm_concat_dim, hidden_dim)
        self.seasonal_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )
        
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1)
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 8 + hidden_dim // 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        if self.training and x.requires_grad:
            short_out = checkpoint.checkpoint(self.short_term, x, use_reentrant=False)[0]
        else:
            short_out, _ = self.short_term(x)
        if seq_len > 2:
            medium_input = x[:, ::2, :]
            medium_out, _ = self.medium_term(medium_input)
            if medium_out.size(1) != seq_len:
                medium_out = F.interpolate(
                    medium_out.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
        else:
            medium_out = short_out
        
        if seq_len > 4:
            long_input = x[:, ::4, :]
            long_out, _ = self.long_term(long_input)
            if long_out.size(1) != seq_len:
                long_out = F.interpolate(
                    long_out.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
        else:
            long_out = short_out
        combined_lstm = torch.cat([short_out, medium_out, long_out], dim=-1)
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        combined_lstm = combined_lstm + pos_enc
        combined_lstm = self.lstm_projection(combined_lstm)
        
        if self.training and x.requires_grad:
            transformer_out = checkpoint.checkpoint(self.transformer_encoder, combined_lstm, use_reentrant=False)
        else:
            transformer_out = self.transformer_encoder(combined_lstm)
        
        seasonal_features = self.seasonal_detector(x)
        seasonal_pooled = self.adaptive_pool(seasonal_features.transpose(1, 2)).squeeze(-1)
        if seasonal_pooled.dim() > 2:
            seasonal_pooled = seasonal_pooled.view(seasonal_pooled.size(0), -1)
        
        trend_features = self.trend_extractor(x.transpose(1, 2))
        trend_pooled = self.adaptive_pool(trend_features).squeeze(-1)
        if trend_pooled.dim() > 2:
            trend_pooled = trend_pooled.view(trend_pooled.size(0), -1)
        
        attended, attention_weights = self.temporal_attention(transformer_out, transformer_out, transformer_out)
        attended_pooled = torch.mean(attended, dim=1)
        
        if seasonal_pooled.dim() == 1:
            seasonal_pooled = seasonal_pooled.unsqueeze(1)
        if trend_pooled.dim() == 1:
            trend_pooled = trend_pooled.unsqueeze(1)
        
        seasonal_expanded = seasonal_pooled.expand(-1, attended_pooled.size(1)) if seasonal_pooled.size(1) == 1 else seasonal_pooled
        trend_expanded = trend_pooled.expand(-1, attended_pooled.size(1)) if trend_pooled.size(1) == 1 else trend_pooled
        
        all_features = torch.cat([attended_pooled, seasonal_expanded, trend_expanded], dim=-1)
        
        output = self.fusion_layers(all_features)
        output = self.norm(output)
        
        return output

class AdaptiveTopologyLearning(nn.Module):
    def __init__(self, num_nodes, hidden_dim, chaos_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.chaos_dim = chaos_dim
        self.spatial_dim = hidden_dim // 2
        
        self.node_encoder = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.chaos_encoder = nn.Sequential(
            nn.Linear(chaos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )
        
        self.edge_importance = nn.Sequential(
            nn.Linear(hidden_dim * 2 + chaos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.graph_conv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.adjacency_projection = None
        self.adjacency_predictor_main = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, num_nodes),
            nn.Sigmoid()
        )
        self.edge_temperature = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, node_features, chaos_features):
        device = node_features.device
        batch_size = 1
        if node_features.size(-1) != self.spatial_dim:
            if not hasattr(self, 'node_input_projection') or self.node_input_projection is None:
                self.node_input_projection = nn.Linear(
                    node_features.size(-1), self.spatial_dim
                ).to(device)
            node_features = self.node_input_projection(node_features)
        
        encoded_nodes = self.node_encoder(node_features)
        encoded_chaos = self.chaos_encoder(chaos_features)
        
        nodes_expanded = encoded_nodes.unsqueeze(0)
        local_attended, _ = self.local_attention(nodes_expanded, nodes_expanded, nodes_expanded)
        global_attended, _ = self.global_attention(nodes_expanded, nodes_expanded, nodes_expanded)
        
        local_attended = local_attended.squeeze(0)
        global_attended = global_attended.squeeze(0)
        
        combined_nodes = (local_attended + global_attended + encoded_nodes) / 3
        static_adj = torch.ones(self.num_nodes, self.num_nodes, device=device) - torch.eye(self.num_nodes, device=device)
        similarity_matrix = torch.mm(combined_nodes, combined_nodes.t())
        similarity_matrix = torch.sigmoid(similarity_matrix) - torch.eye(self.num_nodes, device=device)
  
        distance_matrix = torch.cdist(combined_nodes, combined_nodes, p=2)
        max_dist = distance_matrix.max()
        normalized_distances = distance_matrix / (max_dist + 1e-8)
        distance_weights = self.distance_mlp(normalized_distances.unsqueeze(-1)).squeeze(-1)
        distance_adj = distance_weights * (1 - torch.eye(self.num_nodes, device=device))
        
        chaos_expanded = encoded_chaos.unsqueeze(0).expand(self.num_nodes, -1)
        node_chaos_concat = torch.cat([combined_nodes, chaos_expanded], dim=-1)
        
        if self.adjacency_projection is None:
            input_dim = node_chaos_concat.size(-1)
            self.adjacency_projection = nn.Linear(input_dim, self.hidden_dim).to(device)

        projected_input = self.adjacency_projection(node_chaos_concat)
        chaos_adj_weights = self.adjacency_predictor_main(projected_input)
        chaos_adj = chaos_adj_weights * (1 - torch.eye(self.num_nodes, device=device))
        
        combined_adj = (0.3 * static_adj + 
                       0.25 * similarity_matrix + 
                       0.25 * distance_adj + 
                       0.2 * chaos_adj)
        edge_scores = torch.ones_like(combined_adj, device=device)

        if self.num_nodes <= 50:  
            i_indices, j_indices = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1)
            valid_mask = combined_adj[i_indices, j_indices] > 0.1
            
            if valid_mask.any():
                valid_i = i_indices[valid_mask]
                valid_j = j_indices[valid_mask]
                
                node_i_features = combined_nodes[valid_i]
                node_j_features = combined_nodes[valid_j]
                
                if node_i_features.size(1) < self.hidden_dim:
                    pad_size = self.hidden_dim - node_i_features.size(1)
                    padding = torch.zeros(node_i_features.size(0), pad_size, device=device)
                    node_i_features = torch.cat([node_i_features, padding], dim=1)
                    node_j_features = torch.cat([node_j_features, padding], dim=1)
                
                chaos_expanded_batch = chaos_features.unsqueeze(0).expand(len(valid_i), -1)
                edge_inputs = torch.cat([node_i_features, node_j_features, chaos_expanded_batch], dim=-1)
                
                edge_importances = self.edge_importance(edge_inputs).squeeze(-1)
                
                edge_scores[valid_i, valid_j] = edge_importances
                edge_scores[valid_j, valid_i] = edge_importances
        else:
            num_samples = min(self.num_nodes * 5, 200)
            sample_indices = torch.randperm(self.num_nodes * self.num_nodes)[:num_samples]
            
            for idx in sample_indices:
                i = idx // self.num_nodes
                j = idx % self.num_nodes
                if i != j and combined_adj[i, j] > 0.1:
                    node_i = combined_nodes[i]
                    node_j = combined_nodes[j]
                    
                    if node_i.size(0) < self.hidden_dim:
                        padding = torch.zeros(self.hidden_dim - node_i.size(0), device=device)
                        node_i = torch.cat([node_i, padding])
                        node_j = torch.cat([node_j, padding])
                    
                    edge_input = torch.cat([node_i, node_j, chaos_features])
                    importance = self.edge_importance(edge_input.unsqueeze(0)).squeeze()
                    edge_scores[i, j] = importance
                    edge_scores[j, i] = importance
        
        temperature = torch.sigmoid(self.edge_temperature) + 0.1
        edge_scores = torch.softmax(edge_scores / temperature, dim=-1)
        
        final_adj = combined_adj * edge_scores
        final_adj = (final_adj + final_adj.t()) / 2
        final_adj = final_adj + torch.eye(self.num_nodes, device=device) * 0.5
        
        row_sums = final_adj.sum(dim=1, keepdim=True)
        final_adj = final_adj / (row_sums + 1e-8)
        
        return final_adj

class SpatialProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
    def forward(self, x):
        return self.layers(x)

class UncertaintyPredictor(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_heads=5, num_nodes=None, prediction_horizon=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes or 207  
        self.prediction_horizon = prediction_horizon or 12  
        self.final_output_dim = self.num_nodes * self.prediction_horizon * output_dim
        
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.final_output_dim)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.final_output_dim)
            ),
            self._create_attention_head(hidden_dim, self.final_output_dim),
            self._create_bayesian_head(hidden_dim, self.final_output_dim),
            self._create_ensemble_head(hidden_dim, self.final_output_dim)
        ])
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.final_output_dim),
            nn.Softplus()
        )
        
        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.final_output_dim),
            nn.Softplus()
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim + self.final_output_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.ensemble_weights = nn.Sequential(
            nn.Linear(hidden_dim, num_heads),
            nn.Softmax(dim=-1)
        )
        
        self.meta_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim + self.final_output_dim * num_heads, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.final_output_dim),
            nn.Softplus()
        )
        
    def _create_attention_head(self, hidden_dim, output_dim):
        class AttentionHead(nn.Module):
            def __init__(self, hidden_dim, output_dim):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                self.output_proj = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x_expanded = x.unsqueeze(1)  
                attended, _ = self.attention(x_expanded, x_expanded, x_expanded)
                attended = attended.squeeze(1)  
                return self.output_proj(attended)
                
        return AttentionHead(hidden_dim, output_dim)
    
    def _create_bayesian_head(self, hidden_dim, output_dim):
        class BayesianLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.mean = nn.Linear(in_features, out_features)
                self.log_var = nn.Linear(in_features, out_features)
                
            def forward(self, x):
                mean = self.mean(x)
                log_var = self.log_var(x)
                if self.training:
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    return mean + eps * std
                else:
                    return mean
        
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            BayesianLinear(hidden_dim // 2, output_dim)
        )
    
    def _create_ensemble_head(self, hidden_dim, output_dim, num_sub_networks=3):
        sub_networks = []
        for i in range(num_sub_networks):
            sub_network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU() if i % 2 == 0 else nn.ReLU(),
                nn.Dropout(0.1 + 0.05 * i),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            sub_networks.append(sub_network)
        
        class EnsembleHead(nn.Module):
            def __init__(self, sub_networks):
                super().__init__()
                self.sub_networks = nn.ModuleList(sub_networks)
            
            def forward(self, x):
                outputs = []
                for sub_net in self.sub_networks:
                    outputs.append(sub_net(x))
                return torch.stack(outputs).mean(dim=0)
        
        return EnsembleHead(sub_networks)
    
    def forward(self, x):
        batch_size = x.size(0)

        if x.dim() > 2:
            if x.dim() == 3:  
                x_processed = torch.mean(x, dim=1)
            elif x.dim() == 4:  
                x_processed = torch.mean(x, dim=(1, 2))
            else:
                x_flat = x.view(batch_size, -1)
                if x_flat.size(1) != self.hidden_dim:
                    if not hasattr(self, '_input_projection'):
                        self._input_projection = nn.Linear(x_flat.size(1), self.hidden_dim).to(x.device)
                    x_processed = self._input_projection(x_flat)
                else:
                    x_processed = x_flat
        else:
            x_processed = x
        
        predictions = []
        for i, head in enumerate(self.prediction_heads):
            pred = head(x_processed)
            predictions.append(pred)
        
        all_predictions = torch.stack(predictions, dim=0)  
        
        ensemble_weights = self.ensemble_weights(x_processed)  
        ensemble_weights = ensemble_weights.unsqueeze(-1)  
        ensemble_pred = torch.sum(all_predictions.permute(1, 0, 2) * ensemble_weights, dim=1)  
        ensemble_pred_reshaped = ensemble_pred.view(batch_size, self.prediction_horizon, self.num_nodes)
        
        prediction_variance = torch.var(all_predictions, dim=0)  
        prediction_variance = prediction_variance.view(batch_size, self.prediction_horizon, self.num_nodes)
        
        aleatoric_uncertainty = self.aleatoric_head(x_processed)
        aleatoric_uncertainty = aleatoric_uncertainty.view(batch_size, self.prediction_horizon, self.num_nodes)
        
        epistemic_uncertainty = self.epistemic_head(x_processed)
        epistemic_uncertainty = epistemic_uncertainty.view(batch_size, self.prediction_horizon, self.num_nodes)
        
        meta_input = torch.cat([x_processed, all_predictions.permute(1, 0, 2).reshape(batch_size, -1)], dim=-1)
        meta_uncertainty = self.meta_uncertainty(meta_input)
        meta_uncertainty = meta_uncertainty.view(batch_size, self.prediction_horizon, self.num_nodes)
        
        confidence_input = torch.cat([x_processed, ensemble_pred], dim=-1)
        prediction_confidence = self.confidence_estimator(confidence_input)  
        prediction_confidence = prediction_confidence.unsqueeze(-1) 
        prediction_confidence = prediction_confidence.expand(batch_size, self.prediction_horizon, self.num_nodes)
        
        total_uncertainty = (prediction_variance + 
                           aleatoric_uncertainty + 
                           epistemic_uncertainty + 
                           meta_uncertainty) / 4
        
        scaled_uncertainty = total_uncertainty * (1 - prediction_confidence + 0.1)
        
        uncertainty_dict = {
            'total': scaled_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'epistemic': epistemic_uncertainty,
            'prediction_variance': prediction_variance,
            'meta': meta_uncertainty,
            'confidence': prediction_confidence
        }
        
        individual_preds = []
        for pred in predictions:
            pred_reshaped = pred.view(batch_size, self.prediction_horizon, self.num_nodes)
            individual_preds.append(pred_reshaped)
        
        return ensemble_pred_reshaped, scaled_uncertainty, individual_preds, uncertainty_dict

class ChaosAwareTrafficPredictor(nn.Module):
    def __init__(self, 
                 num_nodes,
                 node_feature_dim, 
                 sequence_length, 
                 hidden_dim=64,
                 chaos_dim=16,
                 output_dim=1,
                 num_heads=4,
                 dropout=0.2,
                 noise_std=0.01,
                 use_checkpointing=False,
                 prediction_horizon=12):  
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.chaos_dim = chaos_dim
        self.output_dim = output_dim
        self.noise_std = noise_std
        self.use_checkpointing = use_checkpointing
        self.prediction_horizon = prediction_horizon
    
        self.chaos_analyzer = ChaosAnalyzer()
        self.temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.spatial_projection = None  
        self.spatial_main = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        self.topology_learner = AdaptiveTopologyLearning(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            chaos_dim=chaos_dim
        )
        
        self.chaos_attention = ChaosAwareAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            chaos_dim=chaos_dim,
            dropout=dropout
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.chaos_feature_enhancer = nn.Sequential(
            nn.Linear(chaos_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
  
        self.predictor = UncertaintyPredictor(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            prediction_horizon=prediction_horizon
        )
        
        self.chaos_regularizer = nn.Sequential(
            nn.Linear(chaos_dim, 1),
            nn.Sigmoid()
        )
        
    def add_feature_noise(self, features, training=True):
        if training and self.noise_std > 0:
            noise = torch.randn_like(features) * self.noise_std
            return features + noise
        return features
    
    def _process_spatial_features(self, spatial_features):
        if self.spatial_projection is None:
            input_dim = spatial_features.size(-1)
            self.spatial_projection = nn.Linear(
                input_dim, self.hidden_dim
            ).to(spatial_features.device)
        
        projected = self.spatial_projection(spatial_features)
        processed = self.spatial_main(projected)
        return processed
        
    def extract_chaos_features(self, temporal_data):
        if temporal_data.dim() == 3:
            batch_size, seq_len, feature_dim = temporal_data.size()
            chaos_features = torch.zeros(batch_size, self.chaos_dim, 
                                       device=temporal_data.device, dtype=torch.float32)
            
            for b in range(batch_size):
                flow_sequence = temporal_data[b, :, 0].cpu().numpy().flatten()
                features = self.chaos_analyzer.extract_chaos_features(flow_sequence)
                chaos_features[b, :] = torch.tensor(
                    features, device=temporal_data.device, dtype=torch.float32
                )
                
        elif temporal_data.dim() == 4:
            batch_size, seq_len, num_nodes, feature_dim = temporal_data.size()
            chaos_features = torch.zeros(batch_size, self.chaos_dim, 
                                       device=temporal_data.device, dtype=torch.float32)
            
            for b in range(batch_size):
                flow_sequence = temporal_data[b, :, :, 0].cpu().numpy().flatten()
                features = self.chaos_analyzer.extract_chaos_features(flow_sequence)
                chaos_features[b, :] = torch.tensor(
                    features, device=temporal_data.device, dtype=torch.float32
                )
        else:
            raise ValueError(f"Unsupported temporal_data shape: {temporal_data.shape}")
        
        return chaos_features
    
    def forward(self, temporal_data, spatial_data=None, chaos_features=None):
        if temporal_data.dim() == 3:
            batch_size, seq_len, feature_dim = temporal_data.size()
            num_nodes = 1
            temporal_data = temporal_data.unsqueeze(2)
        elif temporal_data.dim() == 4:
            batch_size, seq_len, num_nodes, feature_dim = temporal_data.size()
        else:
            raise ValueError(f"Unsupported temporal_data shape: {temporal_data.shape}")
        
        if chaos_features is None:
            chaos_features = self.extract_chaos_features(temporal_data)
        
        chaos_features = self.add_feature_noise(chaos_features, self.training)
        temporal_reshaped = temporal_data.view(batch_size * num_nodes, seq_len, feature_dim)
        if self.use_checkpointing and self.training:
            temporal_features = checkpoint.checkpoint(self.temporal_encoder, temporal_reshaped, use_reentrant=False)
        else:
            temporal_features = self.temporal_encoder(temporal_reshaped)
        temporal_features = temporal_features.view(batch_size, num_nodes, -1)
        
        if spatial_data is not None:
            if hasattr(spatial_data, 'x'):
                spatial_raw_features = spatial_data.x
            else:
                spatial_raw_features = spatial_data
        else:
            spatial_raw_features = temporal_data.mean(dim=1).view(num_nodes, -1)
            
            temporal_std = temporal_data.std(dim=1).view(num_nodes, -1)
            temporal_max = temporal_data.max(dim=1)[0].view(num_nodes, -1)
            temporal_min = temporal_data.min(dim=1)[0].view(num_nodes, -1)
            
            spatial_raw_features = torch.cat([
                spatial_raw_features, temporal_std, temporal_max, temporal_min
            ], dim=-1)
        
        spatial_features = self._process_spatial_features(spatial_raw_features)
        
        if spatial_features.dim() == 2:  
            spatial_features = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)
        elif spatial_features.dim() == 3:  
            if spatial_features.size(0) != batch_size:
                spatial_features = spatial_features.expand(batch_size, -1, -1)
        elif spatial_features.dim() == 4:  
            spatial_features = spatial_features.squeeze(0)
            if spatial_features.size(0) != batch_size:
                spatial_features = spatial_features.expand(batch_size, -1, -1)
        else:
            spatial_features = spatial_features.view(num_nodes, -1)
            spatial_features = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.use_checkpointing and self.training:
            dynamic_topology = checkpoint.checkpoint(
                self.topology_learner, spatial_features[0], chaos_features.mean(dim=0), use_reentrant=False
            )
        else:
            dynamic_topology = self.topology_learner(spatial_features[0], chaos_features.mean(dim=0))
        
        combined_features = torch.cat([temporal_features, spatial_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        chaos_expanded = chaos_features.unsqueeze(1).expand(-1, num_nodes, -1)
        chaos_enhanced_input = torch.cat([chaos_expanded, fused_features], dim=-1)
        chaos_enhanced_features = self.chaos_feature_enhancer(chaos_enhanced_input)
        
        multi_scale_input = torch.cat([fused_features, chaos_enhanced_features], dim=-1)
        multi_scale_features = self.multi_scale_fusion(multi_scale_input)
        attended_features, attention_weights = self.chaos_attention(
            multi_scale_features.unsqueeze(1), 
            chaos_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        ensemble_pred, uncertainty, individual_preds, uncertainty_dict = self.predictor(attended_features)
        
        chaos_scale = self.chaos_regularizer(chaos_features.mean(dim=0))
        ensemble_pred = ensemble_pred * (0.8 + 0.4 * chaos_scale)
        
        final_prediction = ensemble_pred
        final_uncertainty = uncertainty
        auxiliary_outputs = {
            'chaos_features': chaos_features,
            'attention_weights': attention_weights,
            'dynamic_topology': dynamic_topology,
            'uncertainty': final_uncertainty,
            'uncertainty_components': uncertainty_dict,
            'individual_predictions': individual_preds,
            'chaos_scale': chaos_scale,
            'temporal_features': temporal_features,
            'spatial_features': spatial_features,
            'fused_features': multi_scale_features
        }
        
        return final_prediction, auxiliary_outputs
    
    def compute_loss(self, predictions, targets, auxiliary_outputs):
        mse_loss = F.mse_loss(predictions, targets)
        
        uncertainty = auxiliary_outputs['uncertainty']
        uncertainty_components = auxiliary_outputs.get('uncertainty_components', {})
        
        aleatoric_uncertainty = uncertainty_components.get('aleatoric', uncertainty)
        precision = 1.0 / (aleatoric_uncertainty + 1e-6)
        aleatoric_loss = torch.mean(precision * (predictions - targets)**2 + torch.log(aleatoric_uncertainty + 1e-6))
        
        epistemic_uncertainty = uncertainty_components.get('epistemic', uncertainty * 0.1)
        epistemic_loss = torch.mean(epistemic_uncertainty)
        
        confidence = uncertainty_components.get('confidence', torch.ones_like(uncertainty))
        prediction_error = torch.abs(predictions - targets)
        confidence_loss = F.mse_loss(1 - confidence, prediction_error / (prediction_error.max() + 1e-6))
        
        chaos_features = auxiliary_outputs['chaos_features']
        chaos_scale = auxiliary_outputs.get('chaos_scale', torch.ones(1, device=chaos_features.device))
        
        chaos_magnitude = torch.norm(chaos_features, dim=-1).mean()
        chaos_reg = 0.01 * torch.mean(chaos_features**2) * (1 + chaos_scale).mean()
        
        chaos_correlation = torch.corrcoef(chaos_features.t())
        off_diagonal = chaos_correlation - torch.eye(chaos_correlation.size(0), device=chaos_correlation.device)
        chaos_diversity_loss = 0.01 * torch.mean(off_diagonal**2)
        
        dynamic_topology = auxiliary_outputs['dynamic_topology']
        sparsity_loss = 0.01 * torch.mean(dynamic_topology)
        topology_smoothness = 0.005 * torch.mean((dynamic_topology[:-1, :] - dynamic_topology[1:, :])**2)
        
        individual_preds = auxiliary_outputs['individual_predictions']
        diversity_loss = 0.0
        if len(individual_preds) > 1:
            for i in range(len(individual_preds)):
                for j in range(i+1, len(individual_preds)):
                    diversity_loss += F.mse_loss(individual_preds[i], individual_preds[j])
            diversity_loss = -0.05 * diversity_loss / (len(individual_preds) * (len(individual_preds) - 1) / 2)
        
        temporal_features = auxiliary_outputs.get('temporal_features')
        spatial_features = auxiliary_outputs.get('spatial_features')
        feature_reg = 0.0
        if temporal_features is not None:
            feature_reg += 0.001 * torch.mean(temporal_features**2)
        if spatial_features is not None:
            feature_reg += 0.001 * torch.mean(spatial_features**2)
        
        attention_weights = auxiliary_outputs.get('attention_weights')
        attention_entropy_loss = 0.0
        if attention_weights is not None:
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
            target_entropy = torch.log(torch.tensor(attention_weights.size(-1), dtype=torch.float)) * 0.7
            attention_entropy_loss = 0.01 * F.mse_loss(attention_entropy.mean(), target_entropy)
        
        fused_features = auxiliary_outputs.get('fused_features')
        consistency_loss = 0.0
        if fused_features is not None and fused_features.size(0) > 1:
            feature_similarity = torch.matmul(F.normalize(fused_features.view(fused_features.size(0), -1), dim=1),
                                            F.normalize(fused_features.view(fused_features.size(0), -1), dim=1).t())
            pred_similarity = torch.matmul(F.normalize(predictions.view(predictions.size(0), -1), dim=1),
                                         F.normalize(predictions.view(predictions.size(0), -1), dim=1).t())
            consistency_loss = 0.01 * F.mse_loss(feature_similarity, pred_similarity)
        
        uncertainty_weight = 0.1 * (1 + chaos_magnitude.clamp(0, 1))
        
        total_loss = (mse_loss + 
                     uncertainty_weight * aleatoric_loss +
                     0.05 * epistemic_loss +
                     0.1 * confidence_loss +
                     chaos_reg + 
                     chaos_diversity_loss +
                     sparsity_loss + 
                     topology_smoothness +
                     diversity_loss +
                     feature_reg +
                     attention_entropy_loss +
                     consistency_loss)
        
        return total_loss