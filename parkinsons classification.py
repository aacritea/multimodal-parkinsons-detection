"""
Multimodal Parkinson's Disease Detection Pipeline
IEEE Conference Paper - Clean Implementation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ============================================================================
# PART 1: VOICE DATA PREPROCESSING
# ============================================================================

def load_and_preprocess_parkinsons_data(filepath):
    """Load and preprocess UCI Parkinson's Dataset (ID 174)."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
        print("'name' column dropped")
    
    X = df.drop(columns=['status']).values
    y = df['status'].values
    feature_names = df.drop(columns=['status']).columns.tolist()
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Stratified 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Scaler fitted on training data only\n")
    
    return {
        'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled, 'y_train': y_train,
        'y_val': y_val, 'y_test': y_test,
        'scaler': scaler, 'feature_names': feature_names
    }

# ============================================================================
# PART 2: VOICE PYTORCH DATASET
# ============================================================================

class ParkinsonsVoiceDataset(Dataset):
    """Minimal PyTorch Dataset for UCI Parkinson's voice features."""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# PART 3: VOICE MLP ENCODER
# ============================================================================

class VoiceMLPEncoder(nn.Module):
    """MLP Encoder for voice features."""
    
    def __init__(self, input_dim=22, embedding_dim=128, hidden_dim=384, dropout=0.3):
        super(VoiceMLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim), nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

# ============================================================================
# PART 4: GAIT DATASET
# ============================================================================

class PhysioNetGaitDataset(Dataset):
    """PyTorch Dataset for PhysioNet Gait data."""
    
    def __init__(self, data_dir, label_dict, window_size=256, stride=128, normalize=True):
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        
        self.windows = []
        self.labels = []
        self.subject_ids = []
        
        self._load_and_segment_data()
    
    def _load_and_segment_data(self):
        txt_files = sorted(Path(self.data_dir).glob('*.txt'))
        
        if len(txt_files) == 0:
            raise ValueError(f"No .txt files found in {self.data_dir}")
        
        print(f"Found {len(txt_files)} files")
        processed_count = 0
        
        for file_idx, filepath in enumerate(txt_files):
            filename = filepath.stem
            
            if filename not in self.label_dict:
                continue
            
            try:
                data = np.loadtxt(filepath)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                label = self.label_dict[filename]
                num_windows = (len(data) - self.window_size) // self.stride + 1
                
                for i in range(num_windows):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.window_size
                    window = data[start_idx:end_idx, :]
                    
                    if self.normalize:
                        mean = window.mean(axis=0, keepdims=True)
                        std = window.std(axis=0, keepdims=True) + 1e-8
                        window = (window - mean) / std
                    
                    self.windows.append(window)
                    self.labels.append(label)
                    self.subject_ids.append(file_idx)
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} subjects...")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        print(f"\n✓ Total windows created: {len(self.windows)}")
        print(f"✓ Total subjects processed: {len(set(self.subject_ids))}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        
        window_tensor = torch.tensor(window.T, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        subject_id_tensor = torch.tensor(subject_id, dtype=torch.long)
        
        return window_tensor, label_tensor, subject_id_tensor
    
    def get_num_channels(self):
        return self.windows[0].shape[1] if len(self.windows) > 0 else 0

# ============================================================================
# PART 5: GAIT LABEL GENERATION
# ============================================================================

def generate_gait_label_dict(data_dir):
    """Generate labels based on filename patterns (Co=0, Pt=1)."""
    label_dict = {}
    skip_files = ['demographics', 'format', 'SHA256SUMS']
    txt_files = sorted(Path(data_dir).glob('*.txt'))
    
    for filepath in txt_files:
        filename = filepath.stem
        if filename in skip_files:
            continue
        
        if 'Co' in filename:
            label_dict[filename] = 0
        elif 'Pt' in filename:
            label_dict[filename] = 1
    
    num_controls = sum(1 for v in label_dict.values() if v == 0)
    num_patients = sum(1 for v in label_dict.values() if v == 1)
    print(f"Generated labels for {len(label_dict)} files:")
    print(f"  Controls: {num_controls}, Patients: {num_patients}\n")
    
    return label_dict

# ============================================================================
# PART 6: SUBJECT-LEVEL SPLITTING
# ============================================================================

def subject_level_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Perform subject-level train/val/test split."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    subject_ids = np.array(dataset.subject_ids)
    labels = np.array(dataset.labels)
    unique_subjects = np.unique(subject_ids)
    
    subject_labels = []
    for subj_id in unique_subjects:
        subj_label = labels[subject_ids == subj_id][0]
        subject_labels.append(subj_label)
    subject_labels = np.array(subject_labels)
    
    print(f"Total unique subjects: {len(unique_subjects)}")
    print(f"Subject distribution: {np.bincount(subject_labels.astype(int))}")
    
    train_subjects, temp_subjects, train_labels, temp_labels = train_test_split(
        unique_subjects, subject_labels, test_size=(val_ratio + test_ratio),
        stratify=subject_labels, random_state=random_seed
    )
    
    val_proportion = val_ratio / (val_ratio + test_ratio)
    val_subjects, test_subjects, val_labels, test_labels = train_test_split(
        temp_subjects, temp_labels, test_size=(1 - val_proportion),
        stratify=temp_labels, random_state=random_seed
    )
    
    train_subjects = train_subjects.tolist()
    val_subjects = val_subjects.tolist()
    test_subjects = test_subjects.tolist()
    
    train_indices = [i for i, s in enumerate(subject_ids) if s in train_subjects]
    val_indices = [i for i, s in enumerate(subject_ids) if s in val_subjects]
    test_indices = [i for i, s in enumerate(subject_ids) if s in test_subjects]
    
    print(f"\nSubject-level split:")
    print(f"  Train: {len(train_subjects)} subjects, {len(train_indices)} windows")
    print(f"  Val: {len(val_subjects)} subjects, {len(val_indices)} windows")
    print(f"  Test: {len(test_subjects)} subjects, {len(test_indices)} windows")
    
    assert len(set(train_subjects) & set(val_subjects)) == 0
    assert len(set(train_subjects) & set(test_subjects)) == 0
    assert len(set(val_subjects) & set(test_subjects)) == 0
    print("✓ No subject overlap - splits are valid!\n")
    
    return {
        'train_subjects': train_subjects, 'val_subjects': val_subjects,
        'test_subjects': test_subjects, 'train_indices': train_indices,
        'val_indices': val_indices, 'test_indices': test_indices
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MULTIMODAL PARKINSON'S DETECTION PIPELINE")
    print("="*70)
    
    # 1. Load Voice Data
    print("\n[1/3] LOADING VOICE DATA...")
    print("-"*70)
    voice_filepath = '/Users/aakritijain/Desktop/parkinsons research/parkinsons 2/parkinsons.data'
    voice_data = load_and_preprocess_parkinsons_data(voice_filepath)
    
    # 2. Load Gait Data
    print("\n[2/3] LOADING GAIT DATA...")
    print("-"*70)
    gait_dir = '/Users/aakritijain/Desktop/parkinsons research/gait-in-parkinsons-disease-1.0.0'
    label_dict = generate_gait_label_dict(gait_dir)
    gait_dataset = PhysioNetGaitDataset(gait_dir, label_dict, window_size=256, stride=128)
    
    print(f"Gait dataset size: {len(gait_dataset)}")
    print(f"Number of channels: {gait_dataset.get_num_channels()}")
    
    # 3. Subject-level split for gait
    print("\n[3/3] PERFORMING SUBJECT-LEVEL SPLIT...")
    print("-"*70)
    splits = subject_level_split(gait_dataset)
    
    # Create DataLoaders
    gait_train = Subset(gait_dataset, splits['train_indices'])
    gait_val = Subset(gait_dataset, splits['val_indices'])
    gait_test = Subset(gait_dataset, splits['test_indices'])
    
    print("\n" + "="*70)
    print("PIPELINE READY FOR TRAINING")
    print("="*70)
    print("\nDatasets created:")
    print(f"  Voice: {len(voice_data['X_train_scaled'])} train, "
          f"{len(voice_data['X_val_scaled'])} val, {len(voice_data['X_test_scaled'])} test")
    print(f"  Gait: {len(gait_train)} train, {len(gait_val)} val, {len(gait_test)} test windows")

# Add this right after the "PIPELINE READY FOR TRAINING" section and before the encoder initialization

# ============================================================================
# CREATE DATALOADERS
# ============================================================================

def create_simple_dataloaders(voice_data, gait_dataset, gait_splits, batch_size=32):
    """Create separate DataLoaders for voice and gait."""
    
    # Create voice datasets
    train_voice_dataset = ParkinsonsVoiceDataset(
        voice_data['X_train_scaled'], 
        voice_data['y_train']
    )
    val_voice_dataset = ParkinsonsVoiceDataset(
        voice_data['X_val_scaled'], 
        voice_data['y_val']
    )
    test_voice_dataset = ParkinsonsVoiceDataset(
        voice_data['X_test_scaled'], 
        voice_data['y_test']
    )
    
    # Create gait subsets using the split indices
    train_gait_dataset = Subset(gait_dataset, gait_splits['train_indices'])
    val_gait_dataset = Subset(gait_dataset, gait_splits['val_indices'])
    test_gait_dataset = Subset(gait_dataset, gait_splits['test_indices'])
    
    # Create DataLoaders for voice
    train_loader_voice = DataLoader(train_voice_dataset, batch_size=batch_size, shuffle=True)
    val_loader_voice = DataLoader(val_voice_dataset, batch_size=batch_size, shuffle=False)
    test_loader_voice = DataLoader(test_voice_dataset, batch_size=batch_size, shuffle=False)
    
    # Create DataLoaders for gait
    train_loader_gait = DataLoader(train_gait_dataset, batch_size=batch_size, shuffle=True)
    val_loader_gait = DataLoader(val_gait_dataset, batch_size=batch_size, shuffle=False)
    test_loader_gait = DataLoader(test_gait_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n" + "="*70)
    print("DATALOADERS CREATED")
    print("="*70)
    print(f"Batch size: {batch_size}")
    print(f"Voice - Train: {len(train_loader_voice)} batches, Val: {len(val_loader_voice)} batches, Test: {len(test_loader_voice)} batches")
    print(f"Gait - Train: {len(train_loader_gait)} batches, Val: {len(val_loader_gait)} batches, Test: {len(test_loader_gait)} batches")
    print("="*70 + "\n")
    
    return {
        'train_voice': train_loader_voice,
        'train_gait': train_loader_gait,
        'val_voice': val_loader_voice,
        'val_gait': val_loader_gait,
        'test_voice': test_loader_voice,
        'test_gait': test_loader_gait
    }

# Create the dataloaders
dataloaders = create_simple_dataloaders(
    voice_data=voice_data,
    gait_dataset=gait_dataset,
    gait_splits=splits,
    batch_size=16
)

# Now extract them for use in training
train_loader_voice = dataloaders['train_voice']
train_loader_gait = dataloaders['train_gait']
val_loader_voice = dataloaders['val_voice']
val_loader_gait = dataloaders['val_gait']
test_loader_voice = dataloaders['test_voice']
test_loader_gait = dataloaders['test_gait']

##### ============================================================================
import torch
import torch.nn as nn

class GaitCNNEncoder(nn.Module):
    """
    1D CNN Encoder for PhysioNet Gait time-series windows.
    
    Projects multi-channel VGRF time-series into a 128-dimensional embedding space
    for multimodal fusion with voice features.
    """
    
    def __init__(self, in_channels=19, embedding_dim=128, dropout=0.3):
        """
        Parameters:
        -----------
        in_channels : int
            Number of input channels (default: 19 for PhysioNet Gait)
        embedding_dim : int
            Dimension of output embedding (default: 128)
        dropout : float
            Dropout probability for regularization (default: 0.3)
        """
        super(GaitCNNEncoder, self).__init__()
        
        # Convolutional feature extraction layers
        self.conv_layers = nn.Sequential(
            # Layer 1: Extract low-level temporal features
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 2: Capture mid-level patterns
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 3: Extract high-level representations
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 4: Deep feature refinement
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Global average pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer to project to embedding dimension
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, time_steps)
            e.g., (32, 19, 256)
        
        Returns:
        --------
        torch.Tensor
            Embedding tensor of shape (batch_size, 128)
        """
        # Convolutional feature extraction
        # Input: (batch, 19, 256)
        x = self.conv_layers(x)  # Output: (batch, 256, 16)
        
        # Global average pooling
        x = self.global_pool(x)  # Output: (batch, 256, 1)
        
        # Flatten
        x = x.squeeze(-1)  # Output: (batch, 256)
        
        # Project to embedding dimension
        embedding = self.fc(x)  # Output: (batch, 128)
        
        return embedding


# Example usage (commented out):
# # Initialize encoder
gait_encoder = GaitCNNEncoder(in_channels=19, embedding_dim=128)
# 
# # Test with dummy input (batch_size=32, channels=19, time_steps=256)
batch_size = 32
dummy_input = torch.randn(batch_size, 19, 256)
embedding = gait_encoder(dummy_input)
# 
print(f"Input shape: {dummy_input.shape}")
print(f"Embedding shape: {embedding.shape}")  # Should be (32, 128)
# 
# # Count parameters
total_params = sum(p.numel() for p in gait_encoder.parameters())
trainable_params = sum(p.numel() for p in gait_encoder.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

class AttentionFusion(nn.Module):
    """
    Attention-based multimodal fusion module.
    
    Learns to dynamically weight voice and gait embeddings on a per-sample basis,
    allowing the model to emphasize the most informative modality for each input.
    """
    
    def __init__(self, embedding_dim=128, hidden_dim=64):
        """
        Parameters:
        -----------
        embedding_dim : int
            Dimension of input embeddings (default: 128)
        hidden_dim : int
            Dimension of hidden layer for attention computation (default: 64)
        """
        super(AttentionFusion, self).__init__()
        
        # Attention network: learns importance scores for each modality
        # Takes concatenated embeddings and outputs 2 attention scores
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # 2 scores: one for voice, one for gait
        )
        
        # Softmax to normalize attention weights to sum to 1
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, voice_embedding, gait_embedding):
        """
        Compute attention-weighted fusion of voice and gait embeddings.
        
        Parameters:
        -----------
        voice_embedding : torch.Tensor
            Voice features of shape (batch_size, 128)
        gait_embedding : torch.Tensor
            Gait features of shape (batch_size, 128)
        
        Returns:
        --------
        fused_embedding : torch.Tensor
            Attention-weighted fused embedding of shape (batch_size, 128)
        attention_weights : torch.Tensor
            Normalized attention weights of shape (batch_size, 2)
            where [:, 0] = voice weight, [:, 1] = gait weight
        """
        batch_size = voice_embedding.size(0)
        
        # Concatenate embeddings for attention computation
        combined = torch.cat([voice_embedding, gait_embedding], dim=1)  # (B, 256)
        
        # Compute attention scores
        attention_scores = self.attention_net(combined)  # (B, 2)
        
        # Normalize to get attention weights (sum to 1 per sample)
        attention_weights = self.softmax(attention_scores)  # (B, 2)
        
        # Extract individual weights
        voice_weight = attention_weights[:, 0].unsqueeze(1)  # (B, 1)
        gait_weight = attention_weights[:, 1].unsqueeze(1)   # (B, 1)
        
        # Apply attention weights to embeddings
        weighted_voice = voice_weight * voice_embedding  # (B, 128)
        weighted_gait = gait_weight * gait_embedding     # (B, 128)
        
        # Fuse by summing weighted embeddings
        fused_embedding = weighted_voice + weighted_gait  # (B, 128)
        
        return fused_embedding, attention_weights


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal Parkinson's disease classifier.
    
    Architecture:
    1. Voice encoder (MLP) → 128-dim embedding
    2. Gait encoder (CNN) → 128-dim embedding
    3. Attention fusion → weighted 128-dim embedding
    4. Classification head → binary prediction
    """
    
    def __init__(self, voice_encoder, gait_encoder, fusion_hidden_dim=64, 
                 classifier_hidden_dim=64, dropout=0.3):
        """
        Parameters:
        -----------
        voice_encoder : VoiceMLPEncoder
            Pre-initialized voice encoder
        gait_encoder : GaitCNNEncoder
            Pre-initialized gait encoder
        fusion_hidden_dim : int
            Hidden dimension for attention computation (default: 64)
        classifier_hidden_dim : int
            Hidden dimension for classification head (default: 64)
        dropout : float
            Dropout probability (default: 0.3)
        """
        super(MultimodalClassifier, self).__init__()
        
        # Modality encoders
        self.voice_encoder = voice_encoder
        self.gait_encoder = gait_encoder
        
        # Attention-based fusion
        self.fusion = AttentionFusion(
            embedding_dim=128, 
            hidden_dim=fusion_hidden_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1)  # Binary classification
        )
    
    def forward(self, voice_input, gait_input, return_attention=False):
        """
        Forward pass through the complete multimodal network.
        
        Parameters:
        -----------
        voice_input : torch.Tensor
            Voice features of shape (batch_size, 22)
        gait_input : torch.Tensor
            Gait windows of shape (batch_size, 19, 256)
        return_attention : bool
            If True, return attention weights for interpretability (default: False)
        
        Returns:
        --------
        logits : torch.Tensor
            Classification logits of shape (batch_size, 1)
        attention_weights : torch.Tensor (optional)
            Attention weights of shape (batch_size, 2) if return_attention=True
        """
        # Extract embeddings from both modalities
        voice_embedding = self.voice_encoder(voice_input)  # (B, 128)
        gait_embedding = self.gait_encoder(gait_input)     # (B, 128)
        
        # Fuse with learned attention
        fused_embedding, attention_weights = self.fusion(
            voice_embedding, 
            gait_embedding
        )
        
        # Classification
        logits = self.classifier(fused_embedding)  # (B, 1)
        
        if return_attention:
            return logits, attention_weights
        return logits


# Example usage (commented out):
# # Initialize encoders
voice_encoder = VoiceMLPEncoder(input_dim=22, embedding_dim=128)
gait_encoder = GaitCNNEncoder(in_channels=19, embedding_dim=128)
# 
# # Initialize multimodal classifier
model = MultimodalClassifier(
    voice_encoder=voice_encoder,
    gait_encoder=gait_encoder,
    fusion_hidden_dim=64,
    classifier_hidden_dim=64,
    dropout=0.3
)
# 
# # Test forward pass
batch_size = 32
voice_input = torch.randn(batch_size, 22)
gait_input = torch.randn(batch_size, 19, 256)
# 
# # Get predictions
logits = model(voice_input, gait_input)
print(f"Logits shape: {logits.shape}")  # Should be (32, 1)
# 
# # Get predictions with attention weights (for interpretability)
logits, attention_weights = model(voice_input, gait_input, return_attention=True)
print(f"Attention weights shape: {attention_weights.shape}")  # Should be (32, 2)
print(f"Sample attention weights (voice, gait): {attention_weights[0]}")
# 
# # Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ClassificationHead(nn.Module):
    """
    Binary classification head for Parkinson's disease prediction.
    
    Takes fused embedding and outputs logits for BCEWithLogitsLoss.
    """
    
    def __init__(self, input_dim=128, hidden_dim=64, dropout=0.3):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input fused embedding (default: 128)
        hidden_dim : int
            Dimension of hidden layer (default: 64)
        dropout : float
            Dropout probability (default: 0.3)
        """
        super(ClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            # Hidden layer for additional non-linear transformation
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer (logits for BCEWithLogitsLoss)
            nn.Linear(hidden_dim, 1)  # Single logit for binary classification
        )
    
    def forward(self, fused_embedding):
        """
        Forward pass through classification head.
        
        Parameters:
        -----------
        fused_embedding : torch.Tensor
            Fused embedding of shape (batch_size, 128)
        
        Returns:
        --------
        logits : torch.Tensor
            Classification logits of shape (batch_size, 1)
            Note: Returns raw logits (NOT probabilities)
        """
        logits = self.classifier(fused_embedding)  # (B, 1)
        return logits


class MultimodalClassifierWithHead(nn.Module):
    """
    Complete multimodal Parkinson's disease classifier with classification head.
    
    Architecture:
    1. Voice encoder (MLP) → 128-dim embedding
    2. Gait encoder (CNN) → 128-dim embedding
    3. Attention fusion → weighted 128-dim embedding
    4. Classification head → binary logits
    """
    
    def __init__(self, voice_encoder, gait_encoder, fusion_hidden_dim=64, 
                 classifier_hidden_dim=64, dropout=0.3):
        """
        Parameters:
        -----------
        voice_encoder : VoiceMLPEncoder
            Pre-initialized voice encoder
        gait_encoder : GaitCNNEncoder
            Pre-initialized gait encoder
        fusion_hidden_dim : int
            Hidden dimension for attention computation (default: 64)
        classifier_hidden_dim : int
            Hidden dimension for classification head (default: 64)
        dropout : float
            Dropout probability (default: 0.3)
        """
        super(MultimodalClassifierWithHead, self).__init__()
        
        # Modality encoders
        self.voice_encoder = voice_encoder
        self.gait_encoder = gait_encoder
        
        # Attention-based fusion (from previous artifact)
        self.fusion = AttentionFusion(
            embedding_dim=128, 
            hidden_dim=fusion_hidden_dim
        )
        
        # Classification head
        self.classifier_head = ClassificationHead(
            input_dim=128,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout
        )
    
    def forward(self, voice_input, gait_input, return_attention=False):
        """
        Forward pass through the complete multimodal network.
        
        Parameters:
        -----------
        voice_input : torch.Tensor
            Voice features of shape (batch_size, 22)
        gait_input : torch.Tensor
            Gait windows of shape (batch_size, 19, 256)
        return_attention : bool
            If True, return attention weights for interpretability (default: False)
        
        Returns:
        --------
        logits : torch.Tensor
            Classification logits of shape (batch_size, 1)
            Raw logits suitable for BCEWithLogitsLoss
        attention_weights : torch.Tensor (optional)
            Attention weights of shape (batch_size, 2) if return_attention=True
        """
        # Extract embeddings from both modalities
        voice_embedding = self.voice_encoder(voice_input)  # (B, 128)
        gait_embedding = self.gait_encoder(gait_input)     # (B, 128)
        
        # Fuse with learned attention
        fused_embedding, attention_weights = self.fusion(
            voice_embedding, 
            gait_embedding
        )
        
        # Classification (outputs logits, not probabilities)
        logits = self.classifier_head(fused_embedding)  # (B, 1)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_proba(self, voice_input, gait_input):
        """
        Get probability predictions (for evaluation).
        
        Parameters:
        -----------
        voice_input : torch.Tensor
            Voice features
        gait_input : torch.Tensor
            Gait windows
        
        Returns:
        --------
        probabilities : torch.Tensor
            Predicted probabilities of shape (batch_size, 1)
        """
        logits = self.forward(voice_input, gait_input)
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        return probabilities


# AttentionFusion module (needed for the classifier)
import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Attention-based multimodal fusion module with modality dropout.
    
    Learns to dynamically weight voice and gait embeddings on a per-sample basis,
    with optional modality dropout during training for robustness.
    """
    
    def __init__(self, embedding_dim=128, hidden_dim=64, modality_dropout=0.0):
        """
        Parameters:
        -----------
        embedding_dim : int
            Dimension of input embeddings (default: 128)
        hidden_dim : int
            Dimension of hidden layer for attention computation (default: 64)
        modality_dropout : float
            Probability of dropping each modality during training (default: 0.0)
        """
        super(AttentionFusion, self).__init__()
        
        self.modality_dropout = modality_dropout
        
        # Attention network: learns importance scores for each modality
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # 2 scores: one for voice, one for gait
        )
        
        # Softmax to normalize attention weights to sum to 1
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, voice_embedding, gait_embedding):
        """
        Compute attention-weighted fusion with optional modality dropout.
        """
        batch_size = voice_embedding.size(0)
        device = voice_embedding.device
    
    # Apply modality dropout during training
        if self.training and self.modality_dropout > 0:
        # Create masks
            voice_mask = (torch.rand(batch_size, 1, device=device) > self.modality_dropout).float()
            gait_mask = (torch.rand(batch_size, 1, device=device) > self.modality_dropout).float()
        
        # Ensure at least one modality is active per sample
            both_dropped = (voice_mask.squeeze() == 0) & (gait_mask.squeeze() == 0)
            if both_dropped.any():
            # Randomly choose voice or gait for samples where both were dropped
                rescue_voice = torch.rand(batch_size, device=device) > 0.5
                voice_mask[both_dropped] = rescue_voice[both_dropped].float().unsqueeze(1)
                gait_mask[both_dropped] = (~rescue_voice[both_dropped]).float().unsqueeze(1)
        
        # Apply masks (create new tensors, don't modify in-place)
            voice_embedding_masked = voice_embedding * voice_mask
            gait_embedding_masked = gait_embedding * gait_mask
        else:
            voice_embedding_masked = voice_embedding
            gait_embedding_masked = gait_embedding
    
    # Concatenate embeddings for attention computation
        combined = torch.cat([voice_embedding_masked, gait_embedding_masked], dim=1)  # (B, 256)
    
    # Compute attention scores
        attention_scores = self.attention_net(combined)  # (B, 2)
    
    # Normalize to get attention weights (sum to 1 per sample)
        attention_weights = self.softmax(attention_scores)  # (B, 2)
    
    # Extract individual weights
        voice_weight = attention_weights[:, 0].unsqueeze(1)  # (B, 1)
        gait_weight = attention_weights[:, 1].unsqueeze(1)   # (B, 1)
    
    # Apply attention weights to embeddings
        weighted_voice = voice_weight * voice_embedding_masked  # (B, 128)
        weighted_gait = gait_weight * gait_embedding_masked     # (B, 128)
    
    # Fuse by summing weighted embeddings
        fused_embedding = weighted_voice + weighted_gait  # (B, 128)
    
        return fused_embedding, attention_weights

class MultimodalClassifier(nn.Module):
    """
    Complete multimodal Parkinson's disease classifier with modality dropout.
    
    Architecture:
    1. Voice encoder (MLP) → 128-dim embedding
    2. Gait encoder (CNN) → 128-dim embedding
    3. Attention fusion (with modality dropout) → weighted 128-dim embedding
    4. Classification head → binary prediction
    """
    
    def __init__(self, voice_encoder, gait_encoder, fusion_hidden_dim=64, 
                 classifier_hidden_dim=64, dropout=0.3, modality_dropout=0.0):
        """
        Parameters:
        -----------
        voice_encoder : VoiceMLPEncoder
            Pre-initialized voice encoder
        gait_encoder : GaitCNNEncoder
            Pre-initialized gait encoder
        fusion_hidden_dim : int
            Hidden dimension for attention computation (default: 64)
        classifier_hidden_dim : int
            Hidden dimension for classification head (default: 64)
        dropout : float
            Standard dropout probability (default: 0.3)
        modality_dropout : float
            Probability of dropping each modality during training (default: 0.0)
            Set to 0.1-0.3 for robust training
        """
        super(MultimodalClassifier, self).__init__()
        
        # Modality encoders
        self.voice_encoder = voice_encoder
        self.gait_encoder = gait_encoder
        
        # Attention-based fusion with modality dropout
        self.fusion = AttentionFusion(
            embedding_dim=128, 
            hidden_dim=fusion_hidden_dim,
            modality_dropout=modality_dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1)  # Binary classification
        )
    
    def forward(self, voice_input, gait_input, return_attention=False):
        """
        Forward pass through the complete multimodal network.
        
        Parameters:
        -----------
        voice_input : torch.Tensor
            Voice features of shape (batch_size, 22)
        gait_input : torch.Tensor
            Gait windows of shape (batch_size, 19, 256)
        return_attention : bool
            If True, return attention weights for interpretability (default: False)
        
        Returns:
        --------
        logits : torch.Tensor
            Classification logits of shape (batch_size, 1)
        attention_weights : torch.Tensor (optional)
            Attention weights of shape (batch_size, 2) if return_attention=True
        """
        # Extract embeddings from both modalities
        voice_embedding = self.voice_encoder(voice_input)  # (B, 128)
        gait_embedding = self.gait_encoder(gait_input)     # (B, 128)
        
        # Fuse with learned attention (modality dropout applied here during training)
        fused_embedding, attention_weights = self.fusion(
            voice_embedding, 
            gait_embedding
        )
        
        # Classification
        logits = self.classifier(fused_embedding)  # (B, 1)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_proba(self, voice_input, gait_input):
        """
        Get probability predictions (for evaluation).
        
        Parameters:
        -----------
        voice_input : torch.Tensor
            Voice features
        gait_input : torch.Tensor
            Gait windows
        
        Returns:
        --------
        probabilities : torch.Tensor
            Predicted probabilities of shape (batch_size, 1)
        """
        logits = self.forward(voice_input, gait_input)
        probabilities = torch.sigmoid(logits)
        return probabilities


# Example usage (commented out):
# # Initialize encoders
voice_encoder = VoiceMLPEncoder(input_dim=22, embedding_dim=128)
gait_encoder = GaitCNNEncoder(in_channels=19, embedding_dim=128)
# 
# # Initialize multimodal classifier with modality dropout
model = MultimodalClassifier(
    voice_encoder=voice_encoder,
    gait_encoder=gait_encoder,
    fusion_hidden_dim=64,
    classifier_hidden_dim=64,
    dropout=0.3,
    modality_dropout=0.2  # 20% chance to drop each modality during training
)
# 
# # Training mode: modality dropout is active
model.train()
voice_input = torch.randn(32, 22)
gait_input = torch.randn(32, 19, 256)
logits = model(voice_input, gait_input)
print(f"Training logits: {logits.shape}")

# Evaluation mode: modality dropout is disabled
model.eval()
with torch.no_grad():
    logits, attention = model(voice_input, gait_input, return_attention=True)
    print(f"Eval logits: {logits.shape}")
    print(f"Attention weights: {attention[0]}")
# 
# # Test modality dropout effect
model.train()
attention_weights_list = []
for _ in range(10):
    _, attn = model(voice_input[:4], gait_input[:4], return_attention=True)
    attention_weights_list.append(attn.detach())
print("\nAttention weights across 10 forward passes with modality dropout:")

for i, attn in enumerate(attention_weights_list):
    print(f"  Pass {i+1}: {attn[0]}")  # Shows variation due to dropout


######
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from collections import defaultdict
import time

def train_epoch(model, train_loader_voice, train_loader_gait, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Handles mismatched dataloader lengths by cycling through the shorter one.
    """
    model.train()
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Create iterators
    voice_iter = iter(train_loader_voice)
    
    # Iterate through gait (the longer dataloader)
    for gait_windows, gait_labels, _ in train_loader_gait:
        # Cycle through voice samples
        try:
            voice_features, voice_labels = next(voice_iter)
        except StopIteration:
            # Restart voice iterator when exhausted
            voice_iter = iter(train_loader_voice)
            voice_features, voice_labels = next(voice_iter)
        
        # Ensure batch sizes match (take minimum)
        min_batch = min(voice_features.size(0), gait_windows.size(0))
        voice_features = voice_features[:min_batch]
        gait_windows = gait_windows[:min_batch]
        labels = gait_labels[:min_batch].to(device).unsqueeze(1)
        
        # Move to device
        voice_features = voice_features.to(device)
        gait_windows = gait_windows.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(voice_features, gait_windows)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * min_batch
        
        # Get predictions
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute epoch metrics
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }

def evaluate(model, val_loader_voice, val_loader_gait, criterion, device):
    """
    Evaluate the model on validation or test set.
    Handles mismatched dataloader lengths.
    """
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    voice_iter = iter(val_loader_voice)
    
    with torch.no_grad():
        for gait_windows, gait_labels, _ in val_loader_gait:
            # Cycle through voice samples
            try:
                voice_features, voice_labels = next(voice_iter)
            except StopIteration:
                voice_iter = iter(val_loader_voice)
                voice_features, voice_labels = next(voice_iter)
            
            # Ensure batch sizes match
            min_batch = min(voice_features.size(0), gait_windows.size(0))
            voice_features = voice_features[:min_batch]
            gait_windows = gait_windows[:min_batch]
            labels = gait_labels[:min_batch].to(device).unsqueeze(1)
            
            # Move to device
            voice_features = voice_features.to(device)
            gait_windows = gait_windows.to(device)
            
            # Forward pass
            logits = model(voice_features, gait_windows)
            
            # Compute loss
            loss = criterion(logits, labels)
            running_loss += loss.item() * min_batch
            
            # Get predictions and probabilities
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'labels': np.array(all_labels)
    }

def compute_detailed_metrics(predictions, labels, probabilities):
    """
    Compute comprehensive evaluation metrics for IEEE paper.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Binary predictions (0 or 1)
    labels : numpy.ndarray
        Ground truth labels (0 or 1)
    probabilities : numpy.ndarray
        Predicted probabilities for positive class
    
    Returns:
    --------
    dict: Comprehensive metrics
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    probabilities = probabilities.flatten()
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)  # Sensitivity
    metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
    
    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(labels, probabilities)
    except ValueError:
        metrics['auc_roc'] = 0.0  # Handle cases with only one class
    
    # Confusion matrix values
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    return metrics

# Calculate class weights from your training data
train_labels_gait = [gait_dataset.labels[i] for i in splits['train_indices']]
n_controls = sum(1 for l in train_labels_gait if l == 0)
n_patients = sum(1 for l in train_labels_gait if l == 1)

pos_weight = torch.tensor([n_controls / n_patients])
print(f"Positive class weight: {pos_weight.item():.3f}")

# Then modify the criterion in train_multimodal_model function:
# Change: criterion = nn.BCEWithLogitsLoss()
# To: criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

def train_multimodal_model(model, train_loader_voice, train_loader_gait, val_loader_voice, val_loader_gait, num_epochs=50, learning_rate=1e-3, weight_decay=1e-4, device='cpu', patience=10, verbose=True):
    """
    Complete training pipeline with early stopping.
    
    Parameters:
    -----------
    model : MultimodalClassifier
        The multimodal model to train
    train_loader_voice : DataLoader
        Voice training data
    train_loader_gait : DataLoader
        Gait training data
    val_loader_voice : DataLoader
        Voice validation data
    val_loader_gait : DataLoader
        Gait validation data
    num_epochs : int
        Maximum number of training epochs (default: 50)
    learning_rate : float
        Learning rate for Adam optimizer (default: 1e-3)
    weight_decay : float
        L2 regularization weight (default: 1e-4)
    device : str
        Device to train on (default: 'cpu')
    patience : int
        Early stopping patience (default: 10)
    verbose : bool
        Print progress (default: True)
    
    Returns:
    --------
    dict: Training history and best model state
    """
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    if verbose:
        print("="*70)
        print("TRAINING MULTIMODAL PARKINSON'S CLASSIFIER")
        print("="*70)
        print(f"Device: {device}")
        print(f"Epochs: {num_epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}")
        print(f"Early stopping patience: {patience}")
        print("-"*70)
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader_voice, train_loader_gait, criterion, optimizer, device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader_voice, val_loader_gait, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        epoch_time = time.time() - start_time
        
        # Print progress
        if verbose:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            if verbose:
                print(f"  → New best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nLoaded best model (val_loss: {best_val_loss:.4f})")
    
    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss
    }


# Example usage (commented out):
# # Initialize model
voice_encoder = VoiceMLPEncoder(input_dim=22, embedding_dim=128)
gait_encoder = GaitCNNEncoder(in_channels=19, embedding_dim=128)
model = MultimodalClassifier(voice_encoder, gait_encoder, modality_dropout=0.2)
# 
# # Train model
# Replace your current train_multimodal_model call with:
results = train_multimodal_model(
    model=model,
    train_loader_voice=train_loader_voice,
    train_loader_gait=train_loader_gait,
    val_loader_voice=val_loader_voice,
    val_loader_gait=val_loader_gait,
    num_epochs=50,
    learning_rate=5e-4,  # Changed from 1e-3 to 5e-4 (lower, more stable)
    weight_decay=1e-4,
    device='cpu',
    patience=15,  # Increased from 10 to 15
    verbose=True
)
# 
# # Get trained model
trained_model = results['model']
history = results['history']
# 
# # Evaluate on test set
test_metrics = evaluate(trained_model, test_loader_voice, test_loader_gait, nn.BCEWithLogitsLoss(), device='cpu')
# 
# # Compute detailed metrics for IEEE paper
detailed_metrics = compute_detailed_metrics(
    test_metrics['predictions'],
    test_metrics['labels'],
    test_metrics['probabilities']
)

print("\nTest Set Performance:")
print(f"Accuracy: {detailed_metrics['accuracy']:.4f}")
print(f"Precision: {detailed_metrics['precision']:.4f}")
print(f"Recall/Sensitivity: {detailed_metrics['sensitivity']:.4f}")
print(f"Specificity: {detailed_metrics['specificity']:.4f}")
print(f"F1-Score: {detailed_metrics['f1_score']:.4f}")
print(f"AUC-ROC: {detailed_metrics['auc_roc']:.4f}")
