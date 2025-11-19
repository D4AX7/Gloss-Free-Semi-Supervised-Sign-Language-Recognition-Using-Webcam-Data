# API Documentation

## Core Modules

### 1. Data Collection (`src/collect_data.py`)

```python
from collect_data import DataCollector

# Initialize collector
collector = DataCollector(config_path='configs/config.yaml')

# Collect labeled data
collector.collect_video(
    label='hello',
    num_samples=50,
    output_dir='data',
    is_labeled=True
)
```

### 2. Preprocessing (`src/preprocess.py`)

```python
from preprocess import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(config_path='configs/config.yaml')

# Preprocess dataset
preprocessor.process_dataset(
    input_dir='data/labeled',
    output_file='data/processed_labeled.npz'
)
```

### 3. Dataset (`src/dataset.py`)

```python
from dataset import SignLanguageDataset, get_dataloaders

# Create dataset
dataset = SignLanguageDataset(
    data_path='data/processed_labeled.npz',
    label_mapping_path='data/label_mapping.json',
    is_labeled=True
)

# Get dataloaders
train_loader, val_loader, num_classes = get_dataloaders(
    config=config,
    labeled_path='data/processed_labeled.npz',
    unlabeled_path='data/processed_unlabeled.npz',
    label_mapping_path='data/label_mapping.json'
)

# Iterate
for batch in train_loader:
    frames = batch['frames']  # (B, C, T, H, W)
    hand_lms = batch['hand_landmarks']  # (B, T, 126)
    pose_lms = batch['pose_landmarks']  # (B, T, 132)
    labels = batch['label']  # (B,)
```

### 4. Model (`src/model.py`)

```python
from model import SignLanguageRecognitionModel, create_model
import torch

# Create model
config = {...}
model = create_model(config)

# Forward pass
frames = torch.randn(2, 3, 16, 224, 224)
hand_lms = torch.randn(2, 16, 126)
pose_lms = torch.randn(2, 16, 132)

logits = model(frames, hand_lms, pose_lms)  # (2, num_classes)

# Extract features (for self-supervised learning)
features = model.extract_features(frames, hand_lms, pose_lms)  # (2, hidden_dim)
```

#### Model Components

**CNNEncoder**
```python
from model import CNNEncoder

encoder = CNNEncoder(
    encoder_type='resnet18',  # or 'mobilenet_v2'
    hidden_dim=512,
    pretrained=True
)

# Input: (B, C, T, H, W)
# Output: (B, T, hidden_dim)
```

**TransformerTemporalEncoder**
```python
from model import TransformerTemporalEncoder

temporal_encoder = TransformerTemporalEncoder(
    hidden_dim=512,
    num_heads=8,
    num_layers=4,
    dropout=0.1
)

# Input: (B, T, hidden_dim)
# Output: (B, T, hidden_dim)
```

**LandmarkEncoder**
```python
from model import LandmarkEncoder

hand_encoder = LandmarkEncoder(
    landmark_dim=126,  # 2 hands * 21 landmarks * 3 coords
    hidden_dim=512,
    dropout=0.1
)

pose_encoder = LandmarkEncoder(
    landmark_dim=132,  # 33 landmarks * 4 values
    hidden_dim=512,
    dropout=0.1
)
```

**MultimodalFusion**
```python
from model import MultimodalFusion

fusion = MultimodalFusion(
    hidden_dim=512,
    num_heads=8,
    dropout=0.1
)

# Fuse modalities
fused = fusion(query, key, value)
```

### 5. Training (`src/train.py`)

```python
from train import Trainer

# Create trainer
trainer = Trainer(config, device='cuda')

# Train
trainer.train(train_loader, val_loader)

# Train single epoch
train_loss, train_acc = trainer.train_epoch(train_loader, epoch=0)

# Validate
val_loss, val_acc = trainer.validate(val_loader, epoch=0)
```

### 6. Pseudo-Labeling (`src/pseudo_label.py`)

```python
from pseudo_label import PseudoLabeler

# Create pseudo labeler
labeler = PseudoLabeler(
    config=config,
    model_path='models/best_model.pth',
    device='cuda'
)

# Generate pseudo labels
predictions, confidences = labeler.generate_pseudo_labels(unlabeled_loader)

# Save
labeler.save_pseudo_labels(predictions, confidences, 'data/pseudo_labels.npz')
```

### 7. Inference (`src/infer.py`)

```python
from infer import RealTimeInference

# Create inference engine
inference = RealTimeInference(
    config_path='configs/config.yaml',
    model_path='models/best_model.pth',
    label_mapping_path='data/label_mapping.json',
    device='cuda'
)

# Run real-time inference
inference.run(video_source=0)  # 0 for webcam

# Or process single frame
hand_lms, pose_lms = inference.extract_landmarks(frame)
pred_idx, confidence = inference.predict()
```

### 8. Export (`src/export.py`)

```python
from export import ModelExporter

# Create exporter
exporter = ModelExporter(
    config=config,
    model_path='models/best_model.pth',
    device='cpu'
)

# Export TorchScript
exporter.export_torchscript('models/model_scripted.pt')

# Export ONNX
exporter.export_onnx('models/model.onnx')

# Export quantized
exporter.export_quantized('models/quantized_model.pth')
```

### 9. Utilities (`src/utils.py`)

```python
from utils import (
    AverageMeter,
    accuracy,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    count_parameters,
    EarlyStopping
)

# Average meter
meter = AverageMeter()
meter.update(loss.item(), batch_size)
print(f"Average: {meter.avg}")

# Accuracy
acc1 = accuracy(logits, labels, topk=(1,))

# Save checkpoint
save_checkpoint(
    state={
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    },
    is_best=True,
    checkpoint_dir='models'
)

# Load checkpoint
checkpoint = load_checkpoint(
    checkpoint_path='models/best_model.pth',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)

# Set seed
set_seed(42)

# Count parameters
num_params = count_parameters(model)

# Early stopping
early_stopping = EarlyStopping(patience=10, mode='min')
early_stopping(val_loss)
if early_stopping.early_stop:
    print("Early stopping triggered")
```

## Configuration Schema

```yaml
# config.yaml structure
paths:
  raw_data: str          # Path to raw data
  labeled_data: str      # Path to labeled data
  unlabeled_data: str    # Path to unlabeled data
  models: str            # Path to save models
  logs: str              # Path to logs

data:
  frame_height: int      # Frame height in pixels
  frame_width: int       # Frame width in pixels
  frames_per_clip: int   # Number of frames per clip
  fps: int               # Frames per second
  channels: int          # Number of channels (3 for RGB)

model:
  encoder_type: str      # 'resnet18' or 'mobilenet_v2'
  hidden_dim: int        # Hidden dimension size
  num_heads: int         # Number of attention heads
  num_transformer_layers: int  # Number of transformer layers
  dropout: float         # Dropout rate
  num_classes: int       # Number of output classes

training:
  batch_size: int        # Batch size
  learning_rate: float   # Learning rate
  weight_decay: float    # Weight decay
  epochs: int            # Number of epochs
  warmup_epochs: int     # Number of warmup epochs
  gradient_clip: float   # Gradient clipping value
  label_smoothing: float # Label smoothing
  mixed_precision: bool  # Use mixed precision training

semi_supervised:
  pseudo_label_threshold: float  # Confidence threshold
  unlabeled_batch_ratio: int     # Ratio of unlabeled to labeled
  consistency_weight: float      # Weight for consistency loss
  rampup_epochs: int             # Rampup epochs

self_supervised:
  enabled: bool          # Enable self-supervised pretraining
  pretrain_epochs: int   # Number of pretrain epochs
  temperature: float     # Temperature for contrastive loss
  augmentation_strength: float  # Augmentation strength

inference:
  sliding_window_stride: int    # Stride for sliding window
  confidence_threshold: float   # Minimum confidence threshold
  smoothing_window: int         # Smoothing window size

export:
  onnx_opset_version: int      # ONNX opset version
  quantization_backend: str    # 'qnnpack' or 'fbgemm'

logging:
  tensorboard: bool      # Enable TensorBoard logging
  log_interval: int      # Log every N batches
  checkpoint_interval: int  # Save checkpoint every N epochs
```

## Data Format

### Raw Data (.npz files)
```python
{
    'frames': np.ndarray,      # (T, H, W, 3), uint8
    'hand_landmarks': np.ndarray,  # (T, 2, 63), float32
    'pose_landmarks': np.ndarray,  # (T, 132), float32
    'label': str,              # Label name
    'timestamp': str           # ISO format timestamp (optional)
}
```

### Processed Data (.npz files)
```python
{
    'frames': np.ndarray,      # (N, T, H, W, 3), float32, normalized
    'hand_landmarks': np.ndarray,  # (N, T, 126), float32
    'pose_landmarks': np.ndarray,  # (N, T, 132), float32
    'labels': np.ndarray       # (N,), int64
}
```

### Label Mapping (.json)
```json
{
    "hello": 0,
    "thanks": 1,
    "yes": 2,
    "no": 3,
    ...
}
```

### Pseudo Labels (.npz)
```python
{
    'predictions': np.ndarray,   # (N,), int64
    'confidences': np.ndarray    # (N,), float32
}
```

## Model Checkpoints

```python
{
    'epoch': int,                    # Current epoch
    'model_state_dict': OrderedDict, # Model state
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # Scheduler state (optional)
    'best_acc': float,               # Best accuracy so far
    'config': dict                   # Configuration
}
```

## Example Workflows

### Complete Training Pipeline
```python
import yaml
from dataset import get_dataloaders
from model import create_model
from train import Trainer

# Load config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Get dataloaders
train_loader, val_loader, num_classes = get_dataloaders(
    config,
    'data/processed_labeled.npz',
    'data/processed_unlabeled.npz',
    'data/label_mapping.json'
)

# Update config
config['model']['num_classes'] = num_classes

# Train
trainer = Trainer(config, device='cuda')
trainer.train(train_loader, val_loader)
```

### Inference on Video
```python
from infer import RealTimeInference

inference = RealTimeInference(
    'configs/config.yaml',
    'models/best_model.pth',
    'data/label_mapping.json'
)

inference.run('path/to/video.mp4')
```

### Export All Formats
```python
from export import ModelExporter

exporter = ModelExporter(config, 'models/best_model.pth')
exporter.export_torchscript('models/model_scripted.pt')
exporter.export_onnx('models/model.onnx')
exporter.export_quantized('models/quantized_model.pth')
```
