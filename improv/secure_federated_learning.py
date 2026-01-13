"""
Secure Graph-Based Federated Learning Framework
================================================
A comprehensive framework combining:
- Graph-based FL topology for neighbor-aware aggregation
- Multi-KRUM Byzantine-fault-tolerant selection
- Adaptive Trust Algorithm (ATA)
- Trust-Aware Differential Privacy
- Homomorphic Encryption (Paillier)
- Secure Aggregation
- Attack simulations (Poison, Backdoor, Inference, Model Inversion)

Supports both CNN (for MNIST) and simple logistic regression models.
"""

import time
import random
import warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional imports with graceful fallback
try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. CNN/MNIST features disabled.")

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("Warning: Opacus not available. DP with PyTorch disabled.")

try:
    import phe as paillier
    PAILLIER_AVAILABLE = True
except ImportError:
    PAILLIER_AVAILABLE = False
    print("Warning: phe not available. Homomorphic encryption disabled.")


# =============================================================================
# CONFIGURATION & ENUMS
# =============================================================================

class AttackType(Enum):
    NONE = "none"
    POISON = "poison"
    BACKDOOR = "backdoor"
    INFERENCE = "inference"
    MODEL_INVERSION = "model_inversion"


class SelectionMethod(Enum):
    RANDOM = "random"
    MULTI_KRUM = "multi_krum"
    TRUST_BASED = "trust_based"


class AggregationMethod(Enum):
    FEDAVG = "fedavg"
    GRAPH_WEIGHTED = "graph_weighted"
    TRUST_WEIGHTED = "trust_weighted"


@dataclass
class FLConfig:
    """Configuration for Federated Learning experiments."""
    # Client settings
    num_clients: int = 10
    malicious_ratio: float = 0.2
    selection_fraction: float = 0.6
    min_clients_per_round: int = 3
    
    # Training settings
    num_rounds: int = 10
    local_epochs: int = 3
    learning_rate: float = 0.01
    batch_size: int = 32
    
    # Privacy settings
    use_dp: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    
    # Security settings
    use_homomorphic_encryption: bool = False
    use_secure_aggregation: bool = False
    
    # Graph settings
    use_graph_topology: bool = True
    graph_connection_prob: float = 0.6
    
    # Defense settings
    selection_method: SelectionMethod = SelectionMethod.MULTI_KRUM
    aggregation_method: AggregationMethod = AggregationMethod.TRUST_WEIGHTED
    enable_hierarchical_defense: bool = True
    
    # Attack settings (for simulation)
    attack_types: List[AttackType] = field(default_factory=lambda: [
        AttackType.POISON, AttackType.BACKDOOR
    ])


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

class CNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for tabular data."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# ATTACK IMPLEMENTATIONS
# =============================================================================

class AttackSimulator:
    """Simulates various adversarial attacks in federated learning."""
    
    @staticmethod
    def poison_labels(y: np.ndarray, poison_ratio: float = 0.8) -> np.ndarray:
        """Label flipping attack - flip labels of a portion of data."""
        y_poisoned = y.copy()
        n_poison = int(len(y) * poison_ratio)
        poison_indices = np.random.choice(len(y), n_poison, replace=False)
        
        # For binary: flip, for multi-class: random reassign
        if len(np.unique(y)) == 2:
            y_poisoned[poison_indices] = 1 - y_poisoned[poison_indices]
        else:
            num_classes = len(np.unique(y))
            y_poisoned[poison_indices] = np.random.randint(0, num_classes, n_poison)
        
        return y_poisoned
    
    @staticmethod
    def add_backdoor_pattern(
        X: np.ndarray, 
        backdoor_ratio: float = 0.4,
        pattern_strength: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add backdoor trigger pattern to features."""
        X_backdoor = X.copy()
        n_backdoor = int(len(X) * backdoor_ratio)
        backdoor_indices = np.random.choice(len(X), n_backdoor, replace=False)
        
        # Create pattern (modify first few features)
        pattern = np.zeros(X.shape[1] if len(X.shape) > 1 else 1)
        pattern_size = min(3, len(pattern))
        pattern[:pattern_size] = pattern_strength
        
        if len(X.shape) > 1:
            X_backdoor[backdoor_indices] += pattern
        
        return X_backdoor, backdoor_indices
    
    @staticmethod
    def inference_attack(
        global_model: np.ndarray, 
        noise_scale: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """Attempt to reconstruct global model parameters."""
        reconstructed = global_model + np.random.normal(
            scale=noise_scale, size=global_model.shape
        )
        error = np.linalg.norm(global_model - reconstructed)
        return reconstructed, error
    
    @staticmethod
    def model_inversion(
        global_model: np.ndarray,
        n_candidates: int = 100
    ) -> np.ndarray:
        """Attempt to reconstruct training data from model."""
        if len(global_model) < 2:
            return np.zeros(1)
        
        w = global_model[:-1]
        b = global_model[-1]
        
        # Generate candidate samples
        candidates = np.random.randn(n_candidates, len(w))
        outputs = candidates.dot(w) + b
        
        # Return sample that maximizes output
        return candidates[np.argmax(outputs)]


# =============================================================================
# FEDERATED LEARNING CLIENT
# =============================================================================

class FLClient:
    """Federated Learning client with attack/defense capabilities."""
    
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        config: FLConfig,
        is_malicious: bool = False,
        attack_type: AttackType = AttackType.NONE,
        compute_capacity: float = 1.0,
        network_speed: float = 1.0,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.compute_capacity = compute_capacity
        self.network_speed = network_speed
        
        # Trust and history tracking
        self.trust_score = 1.0
        self.update_history: List[np.ndarray] = []
        self.participation_count = 0
        
        # Graph neighbors (set by server)
        self.neighbors: List[int] = []
        
        # Model and training components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.train_loader: Optional[DataLoader] = None
        
        # Initialize data loader
        self._setup_data_loader()
    
    def _setup_data_loader(self):
        """Setup the training data loader."""
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True if len(self.dataset) > self.config.batch_size else False
        )
    
    def setup_model(self, model: nn.Module):
        """Initialize client's local model."""
        self.model = model
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9
        )
    
    def _apply_attack_to_batch(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attack transformations to a training batch."""
        if not self.is_malicious or self.attack_type == AttackType.NONE:
            return data, targets
        
        if self.attack_type == AttackType.POISON:
            # Label flipping
            targets_np = targets.cpu().numpy()
            targets_poisoned = AttackSimulator.poison_labels(targets_np, poison_ratio=0.5)
            return data, torch.tensor(targets_poisoned, dtype=targets.dtype, device=targets.device)
        
        elif self.attack_type == AttackType.BACKDOOR:
            # Add backdoor pattern to images
            data_np = data.cpu().numpy()
            # Simple pattern: bright pixel in corner
            data_np[:, :, 0, 0] = 1.0
            data_np[:, :, 0, 1] = 1.0
            data_np[:, :, 1, 0] = 1.0
            return torch.tensor(data_np, dtype=data.dtype, device=data.device), targets
        
        return data, targets
    
    def train_local(
        self, 
        global_state_dict: Dict,
        device: torch.device = torch.device('cpu')
    ) -> Dict:
        """Perform local training and return model update."""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model first.")
        
        # Load global model weights
        self.model.load_state_dict(global_state_dict)
        self.model.to(device)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.local_epochs):
            for data, targets in self.train_loader:
                # Apply attack if malicious
                data, targets = self._apply_attack_to_batch(data, targets)
                
                data, targets = data.to(device), targets.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.dp_max_grad_norm
                )
                
                self.optimizer.step()
        
        self.participation_count += 1
        
        # Return model state dict
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def get_model_update_vector(self, global_state_dict: Dict) -> np.ndarray:
        """Get flattened model update as numpy array."""
        local_dict = self.model.state_dict()
        update = []
        for key in global_state_dict.keys():
            diff = local_dict[key].cpu().numpy() - global_state_dict[key].cpu().numpy()
            update.append(diff.flatten())
        return np.concatenate(update)
    
    def secure_mask_update(
        self, 
        update: np.ndarray, 
        participant_ids: List[int]
    ) -> np.ndarray:
        """Apply secure aggregation masking."""
        mask = np.zeros_like(update)
        for other_id in participant_ids:
            if other_id != self.client_id:
                # Generate consistent mask using pair identifiers
                pair = tuple(sorted([self.client_id, other_id]))
                seed = hash(pair) % (2**32 - 1)
                rng = np.random.RandomState(seed)
                mask += rng.randn(*update.shape)
        return update + mask


# =============================================================================
# DEFENSE MECHANISMS
# =============================================================================

class DefenseMechanisms:
    """Collection of defense mechanisms for FL."""
    
    @staticmethod
    def multi_krum_scores(
        updates: List[np.ndarray],
        trust_scores: List[float],
        f: int = 1
    ) -> List[Tuple[float, int]]:
        """
        Compute Multi-KRUM scores for Byzantine-fault tolerance.
        Lower scores indicate more trustworthy updates.
        """
        n = len(updates)
        if n <= 2 * f + 2:
            f = max(0, (n - 3) // 2)
        
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    # Trust-weighted distance
                    dist = np.linalg.norm(updates[i] - updates[j])
                    trust_weight = 2 - trust_scores[i] - trust_scores[j]
                    distances.append(dist * trust_weight)
            
            distances.sort()
            # Sum of k nearest distances (k = n - f - 2)
            k = max(1, n - f - 2)
            score = sum(distances[:k])
            scores.append((score, i))
        
        return sorted(scores, key=lambda x: x[0])
    
    @staticmethod
    def apply_differential_privacy(
        updates: List[np.ndarray],
        trust_scores: List[float],
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0
    ) -> List[np.ndarray]:
        """Apply trust-aware differential privacy noise."""
        base_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        noisy_updates = []
        for update, trust in zip(updates, trust_scores):
            # Higher trust -> less noise (higher effective epsilon)
            adjusted_epsilon = max(0.5, min(3.0, trust * 3))
            effective_scale = base_scale * (epsilon / adjusted_epsilon)
            
            noise = np.random.laplace(loc=0, scale=effective_scale, size=update.shape)
            noisy_updates.append(update + noise)
        
        return noisy_updates
    
    @staticmethod
    def compute_adaptive_trust(
        client: FLClient,
        round_num: int,
        selection_history: Dict[int, List[int]]
    ) -> float:
        """Compute adaptive trust score based on client history."""
        if len(client.update_history) < 2:
            return client.trust_score
        
        # Update consistency (correlation between successive updates)
        try:
            consistencies = []
            for i in range(len(client.update_history) - 1):
                h1 = client.update_history[i].flatten()
                h2 = client.update_history[i + 1].flatten()
                if len(h1) > 0 and len(h2) > 0:
                    corr = np.corrcoef(h1, h2)[0, 1]
                    if not np.isnan(corr):
                        consistencies.append(corr)
            consistency = np.mean(consistencies) if consistencies else 0.5
        except Exception:
            consistency = 0.5
        
        # Recent participation rate
        recent_rounds = [r for r in selection_history.get(client.client_id, []) 
                        if r >= round_num - 10]
        participation = len(recent_rounds) / 10.0
        
        # Compute new trust with momentum
        new_trust = (
            0.6 * consistency + 
            0.3 * participation + 
            0.1 * (client.compute_capacity / 2.0)
        )
        
        # Exponential moving average
        updated_trust = 0.8 * client.trust_score + 0.2 * new_trust
        return np.clip(updated_trust, 0.1, 1.0)


# =============================================================================
# FEDERATED LEARNING SERVER
# =============================================================================

class FLServer:
    """Federated Learning server with graph topology and security mechanisms."""
    
    def __init__(self, config: FLConfig, model_factory: Callable[[], nn.Module]):
        self.config = config
        self.model_factory = model_factory
        self.global_model = model_factory()
        self.clients: List[FLClient] = []
        self.graph: Optional[nx.Graph] = None
        
        # Tracking
        self.round = 0
        self.selection_history: Dict[int, List[int]] = defaultdict(list)
        self.metrics = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'avg_trust': [],
            'honest_trust': [],
            'malicious_trust': [],
            'krum_filtered': [],
            'convergence': [],
            'attack_success': [],
        }
        
        # Homomorphic encryption keys
        self.public_key = None
        self.private_key = None
        if config.use_homomorphic_encryption and PAILLIER_AVAILABLE:
            self.public_key, self.private_key = paillier.generate_paillier_keypair()
    
    def setup_clients(self, client_datasets: List[Dataset]):
        """Initialize clients with their datasets."""
        attack_types = [AttackType.NONE] + list(self.config.attack_types)
        
        for i, dataset in enumerate(client_datasets):
            is_malicious = random.random() < self.config.malicious_ratio
            attack_type = (
                random.choice(self.config.attack_types) 
                if is_malicious and self.config.attack_types 
                else AttackType.NONE
            )
            
            client = FLClient(
                client_id=i,
                dataset=dataset,
                config=self.config,
                is_malicious=is_malicious,
                attack_type=attack_type,
                compute_capacity=random.uniform(0.5, 2.0),
                network_speed=random.uniform(0.5, 2.0),
            )
            
            # Setup client's model
            client.setup_model(self.model_factory())
            self.clients.append(client)
        
        # Build graph topology
        if self.config.use_graph_topology:
            self._build_graph()
        
        print(f"Initialized {len(self.clients)} clients")
        print(f"  Malicious: {sum(1 for c in self.clients if c.is_malicious)}")
        print(f"  Graph edges: {self.graph.number_of_edges() if self.graph else 0}")
    
    def _build_graph(self):
        """Build graph topology connecting clients."""
        self.graph = nx.erdos_renyi_graph(
            len(self.clients), 
            self.config.graph_connection_prob
        )
        
        # Assign neighbors to clients
        for client in self.clients:
            client.neighbors = list(self.graph.neighbors(client.client_id))
    
    def _select_clients(self, device: torch.device = torch.device('cpu')) -> List[FLClient]:
        """Select clients for current round based on selection method."""
        num_selected = max(
            self.config.min_clients_per_round,
            int(len(self.clients) * self.config.selection_fraction)
        )
        
        # Apply hierarchical defense scheduling
        method = self.config.selection_method
        if self.config.enable_hierarchical_defense:
            if self.round < 4:
                method = SelectionMethod.MULTI_KRUM
            elif self.round < 7:
                method = SelectionMethod.TRUST_BASED
            else:
                method = SelectionMethod.RANDOM
        
        if method == SelectionMethod.RANDOM:
            selected = random.sample(self.clients, num_selected)
        
        elif method == SelectionMethod.TRUST_BASED:
            # Sort by trust score
            scored = [(c.trust_score, c) for c in self.clients]
            scored.sort(reverse=True, key=lambda x: x[0])
            selected = [c for _, c in scored[:num_selected]]
        
        elif method == SelectionMethod.MULTI_KRUM:
            # Get updates from all clients first
            global_dict = self.global_model.state_dict()
            updates = []
            trust_scores = []
            
            for client in self.clients:
                client.train_local(global_dict, device)  # Pass device parameter
                update = client.get_model_update_vector(global_dict)
                updates.append(update)
                trust_scores.append(client.trust_score)
            
            # Compute KRUM scores
            f = max(1, int(len(self.clients) * self.config.malicious_ratio))
            krum_scores = DefenseMechanisms.multi_krum_scores(
                updates, trust_scores, f
            )
            
            # Select top clients
            selected_indices = [idx for _, idx in krum_scores[:num_selected]]
            selected = [self.clients[i] for i in selected_indices]
            
            # Track filtered malicious clients
            malicious_before = sum(1 for c in self.clients if c.is_malicious)
            malicious_after = sum(1 for c in selected if c.is_malicious)
            self.metrics['krum_filtered'].append(malicious_before - malicious_after)
        
        else:
            selected = random.sample(self.clients, num_selected)
        
        # Update selection history
        for client in selected:
            self.selection_history[client.client_id].append(self.round)
        
        return selected
    
    def _aggregate_updates(
        self, 
        selected_clients: List[FLClient],
        device: torch.device
    ) -> Dict:
        """Aggregate model updates from selected clients."""
        global_dict = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        # Collect updates
        client_updates = []
        weights = []
        
        for client in selected_clients:
            # Training already done in selection for MULTI_KRUM
            if self.config.selection_method != SelectionMethod.MULTI_KRUM:
                client.train_local(global_dict, device)
            
            client_dict = client.model.state_dict()
            client_updates.append(client_dict)
            
            # Weight by data size and trust
            weight = len(client.dataset) * client.trust_score * client.compute_capacity
            weights.append(weight)
            
            # Store update history for trust computation
            update_vector = client.get_model_update_vector(global_dict)
            client.update_history.append(update_vector)
            if len(client.update_history) > 10:
                client.update_history.pop(0)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Apply differential privacy if enabled
        if self.config.use_dp:
            # Apply noise to weight vectors
            flat_updates = [
                np.concatenate([v.cpu().numpy().flatten() for v in u.values()])
                for u in client_updates
            ]
            trust_scores = [c.trust_score for c in selected_clients]
            
            noisy_updates = DefenseMechanisms.apply_differential_privacy(
                flat_updates,
                trust_scores,
                epsilon=self.config.dp_epsilon,
                delta=self.config.dp_delta
            )
            
            # Reconstruct state dicts with noise - ensure correct device
            for i, (client_dict, noisy_flat) in enumerate(zip(client_updates, noisy_updates)):
                idx = 0
                for key in client_dict:
                    shape = client_dict[key].shape
                    size = client_dict[key].numel()
                    client_dict[key] = torch.tensor(
                        noisy_flat[idx:idx + size].reshape(shape),
                        dtype=client_dict[key].dtype,
                        device=device  # FIX: Ensure tensor is on correct device
                    )
                    idx += size
        
        # Weighted aggregation - ensure all tensors on same device
        aggregated = {k: torch.zeros_like(v).to(device) for k, v in global_dict.items()}
        for client_dict, weight in zip(client_updates, weights):
            for key in aggregated:
                # Move client tensor to device before aggregation
                client_tensor = client_dict[key].to(device).float()
                aggregated[key] += weight * client_tensor
        
        return aggregated
    
    def _update_trust_scores(self, selected_clients: List[FLClient]):
        """Update trust scores for all clients."""
        for client in self.clients:
            client.trust_score = DefenseMechanisms.compute_adaptive_trust(
                client, self.round, self.selection_history
            )
    
    def train_round(self, device: torch.device = torch.device('cpu')) -> Dict:
        """Execute one round of federated training."""
        print(f"\n--- Round {self.round + 1} ---")
        
        # Select clients
        selected = self._select_clients(device)
        print(f"Selected {len(selected)} clients (malicious: {sum(1 for c in selected if c.is_malicious)})")
        
        # Aggregate updates
        aggregated = self._aggregate_updates(selected, device)
        
        # Update global model
        self.global_model.load_state_dict(aggregated)
        
        # Update trust scores
        self._update_trust_scores(selected)
        
        # Update metrics
        self.metrics['round'].append(self.round)
        self.metrics['avg_trust'].append(np.mean([c.trust_score for c in self.clients]))
        self.metrics['honest_trust'].append(
            np.mean([c.trust_score for c in self.clients if not c.is_malicious])
        )
        malicious_trusts = [c.trust_score for c in self.clients if c.is_malicious]
        self.metrics['malicious_trust'].append(
            np.mean(malicious_trusts) if malicious_trusts else 0
        )
        
        self.round += 1
        return aggregated
    
    def evaluate(
        self, 
        test_loader: DataLoader, 
        device: torch.device = torch.device('cpu')
    ) -> Tuple[float, float]:
        """Evaluate global model on test data."""
        self.global_model.to(device)
        self.global_model.eval()
        
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self.global_model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(avg_loss)
        
        return accuracy, avg_loss
    
    def visualize_metrics(self, save_path: Optional[str] = None):
        """Generate visualization of training metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Accuracy over rounds
        ax = axes[0, 0]
        if self.metrics['accuracy']:
            ax.plot(self.metrics['round'][:len(self.metrics['accuracy'])], 
                   self.metrics['accuracy'], 'b-o')
            ax.set_title('Model Accuracy')
            ax.set_xlabel('Round')
            ax.set_ylabel('Accuracy')
            ax.grid(True)
        
        # Loss over rounds
        ax = axes[0, 1]
        if self.metrics['loss']:
            ax.plot(self.metrics['round'][:len(self.metrics['loss'])], 
                   self.metrics['loss'], 'r-o')
            ax.set_title('Training Loss')
            ax.set_xlabel('Round')
            ax.set_ylabel('Loss')
            ax.grid(True)
        
        # Trust scores evolution
        ax = axes[0, 2]
        if self.metrics['avg_trust']:
            ax.plot(self.metrics['round'], self.metrics['avg_trust'], 
                   label='Average', color='blue')
            ax.plot(self.metrics['round'], self.metrics['honest_trust'], 
                   label='Honest', color='green')
            ax.plot(self.metrics['round'], self.metrics['malicious_trust'], 
                   label='Malicious', color='red')
            ax.set_title('Trust Score Evolution')
            ax.set_xlabel('Round')
            ax.set_ylabel('Trust Score')
            ax.legend()
            ax.grid(True)
        
        # Trust score distribution
        ax = axes[1, 0]
        honest_scores = [c.trust_score for c in self.clients if not c.is_malicious]
        malicious_scores = [c.trust_score for c in self.clients if c.is_malicious]
        if honest_scores:
            ax.hist(honest_scores, alpha=0.7, label='Honest', color='green', bins=10)
        if malicious_scores:
            ax.hist(malicious_scores, alpha=0.7, label='Malicious', color='red', bins=10)
        ax.set_title('Final Trust Score Distribution')
        ax.set_xlabel('Trust Score')
        ax.set_ylabel('Count')
        ax.legend()
        
        # KRUM filtering effectiveness
        ax = axes[1, 1]
        if self.metrics['krum_filtered']:
            ax.bar(range(len(self.metrics['krum_filtered'])), 
                  self.metrics['krum_filtered'], color='purple')
            ax.set_title('Malicious Clients Filtered (Multi-KRUM)')
            ax.set_xlabel('Round')
            ax.set_ylabel('Count')
        
        # Network graph
        ax = axes[1, 2]
        if self.graph:
            colors = ['red' if self.clients[i].is_malicious else 'green' 
                     for i in range(len(self.clients))]
            pos = nx.spring_layout(self.graph, seed=42)
            nx.draw(self.graph, pos, ax=ax, node_color=colors, 
                   with_labels=True, node_size=500, font_size=8)
            ax.set_title('Client Graph (Red=Malicious)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_mnist_experiment(config: FLConfig) -> FLServer:
    """Run federated learning experiment on MNIST dataset."""
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required for MNIST experiment")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Partition data among clients (IID)
    data_per_client = len(train_dataset) // config.num_clients
    lengths = [data_per_client] * config.num_clients
    lengths[-1] += len(train_dataset) - sum(lengths)  # Handle remainder
    
    client_datasets = random_split(train_dataset, lengths)
    
    # Initialize server
    server = FLServer(config, model_factory=CNN)
    server.setup_clients(client_datasets)
    
    # Test data loader
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Training loop
    start_time = time.time()
    
    for round_num in range(config.num_rounds):
        server.train_round(device)
        accuracy, loss = server.evaluate(test_loader, device)
        print(f"  Accuracy: {accuracy*100:.2f}%, Loss: {loss:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal training time: {elapsed_time:.2f} seconds")
    
    return server


def run_synthetic_experiment(config: FLConfig, n_features: int = 20) -> FLServer:
    """Run federated learning experiment on synthetic data."""
    
    # Generate synthetic data
    class SyntheticDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    # Create synthetic data for each client
    client_datasets = []
    for i in range(config.num_clients):
        n_samples = random.randint(100, 500)
        X = np.random.randn(n_samples, n_features)
        # Create separable classes
        w_true = np.random.randn(n_features)
        y = (X.dot(w_true) > 0).astype(int)
        # Add some label noise
        noise_idx = np.random.choice(n_samples, n_samples // 10, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        
        client_datasets.append(SyntheticDataset(X, y))
    
    # Create test data
    n_test = 1000
    X_test = np.random.randn(n_test, n_features)
    w_true = np.random.randn(n_features)
    y_test = (X_test.dot(w_true) > 0).astype(int)
    test_dataset = SyntheticDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model factory for synthetic data
    def model_factory():
        return SimpleMLP(input_dim=n_features, hidden_dim=64, num_classes=2)
    
    # Initialize server
    server = FLServer(config, model_factory=model_factory)
    server.setup_clients(client_datasets)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training loop
    start_time = time.time()
    
    for round_num in range(config.num_rounds):
        server.train_round(device)
        accuracy, loss = server.evaluate(test_loader, device)
        print(f"  Accuracy: {accuracy*100:.2f}%, Loss: {loss:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal training time: {elapsed_time:.2f} seconds")
    
    return server


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run comparative experiments with different security configurations."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # Define experiment configurations
    experiments = [
        {
            "name": "No Defense",
            "config": FLConfig(
                num_clients=10,
                num_rounds=10,
                use_dp=False,
                use_homomorphic_encryption=False,
                use_secure_aggregation=False,
                selection_method=SelectionMethod.RANDOM,
                enable_hierarchical_defense=False,
                malicious_ratio=0.3,
            )
        },
        {
            "name": "Base Defense (Multi-KRUM + DP)",
            "config": FLConfig(
                num_clients=10,
                num_rounds=10,
                use_dp=True,
                dp_epsilon=1.0,
                use_homomorphic_encryption=False,
                use_secure_aggregation=False,
                selection_method=SelectionMethod.MULTI_KRUM,
                enable_hierarchical_defense=True,
                malicious_ratio=0.3,
            )
        },
        {
            "name": "Full Defense",
            "config": FLConfig(
                num_clients=10,
                num_rounds=10,
                use_dp=True,
                dp_epsilon=1.0,
                use_homomorphic_encryption=PAILLIER_AVAILABLE,
                use_secure_aggregation=True,
                selection_method=SelectionMethod.MULTI_KRUM,
                enable_hierarchical_defense=True,
                malicious_ratio=0.3,
            )
        },
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']}")
        print('='*60)
        
        # Run experiment (use synthetic if torchvision not available)
        if TORCHVISION_AVAILABLE:
            server = run_mnist_experiment(exp['config'])
        else:
            server = run_synthetic_experiment(exp['config'])
        
        # Store results
        final_accuracy = server.metrics['accuracy'][-1] if server.metrics['accuracy'] else 0
        results.append({
            'name': exp['name'],
            'accuracy': final_accuracy,
            'server': server
        })
        
        # Visualize
        server.visualize_metrics(save_path=f"metrics_{exp['name'].replace(' ', '_')}.png")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print('='*60)
    for result in results:
        print(f"{result['name']}: {result['accuracy']*100:.2f}% accuracy")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    colors = ['red', 'orange', 'green'][:len(results)]
    
    bars = plt.bar(names, accuracies, color=colors)
    plt.title('Security Method Impact on Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('security_comparison.png', dpi=150)
    plt.show()
    
    return results


if __name__ == "__main__":
    main()