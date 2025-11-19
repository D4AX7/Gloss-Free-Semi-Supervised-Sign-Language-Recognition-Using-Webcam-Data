# Gloss-Free, Semi-Supervised Sign Language Recognition Using Webcam Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, production-ready deep learning system for real-time sign language recognition using webcam input, featuring semi-supervised learning with pseudo-labeling and multimodal fusion.

## ✨ Features

- 🎯 **Gloss-Free Learning**: Direct sign-to-text mapping without intermediate gloss representations
- 🔄 **Semi-Supervised Learning**: Leverage large unlabeled datasets with pseudo-labeling
- 🎥 **Multimodal Inputs**: RGB frames + hand landmarks + pose landmarks via MediaPipe
- ⚡ **Real-Time Inference**: <50ms latency with sliding window approach
- 📦 **Model Export**: TorchScript, ONNX, and quantized models for deployment
- 🎓 **Ready to Use**: Complete pipeline from data collection to deployment

## 🏗️ Architecture

```
Input → CNN Encoder (ResNet18/MobileNet) → Transformer Temporal Encoder
     → Landmark Encoders → Attention Fusion → Classifier → Output
```

**Components:**
- **CNN Encoder**: ResNet18 or MobileNetV2 (pretrained on ImageNet)
- **Transformer**: 4-layer temporal encoder with multi-head attention
- **Fusion**: Attention-based multimodal fusion
- **Landmarks**: MediaPipe hand and pose estimation

## 🚀 Quick Start

### Automated Setup (Recommended)

**Windows:**
```bash
quick_start.bat
```

**All Platforms:**
```bash
python setup_and_run.py
```

### Manual Setup

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Generate Synthetic Data (for testing)

```bash
python examples/generate_synthetic_data.py --samples 20
```

#### 3. Preprocess Data

```bash
python src/preprocess.py
```

#### 4. Create Initial Models

```bash
python create_models.py
```

#### 5. Train Model

```bash
python src/train.py
```

#### 6. Pseudo-Label Unlabeled Data (Semi-Supervised)

```bash
python src/pseudo_label.py
```

#### 7. Export Models

```bash
python src/export.py
```

#### 8. Real-Time Inference

```bash
python src/infer.py
```

## 📖 Documentation

- **[INSTALLATION.md](INSTALLATION.md)**: Complete installation and setup guide
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: Detailed training instructions
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**: Complete API reference
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Comprehensive project overview

## 🎯 Collecting Real Data

```bash
# Collect labeled data for different signs
python src/collect_data.py --label hello --samples 50
python src/collect_data.py --label thanks --samples 50
python src/collect_data.py --label yes --samples 50
# ... add more signs

# Collect unlabeled data for semi-supervised learning
python src/collect_data.py --label unknown --samples 100 --unlabeled
```

## 📊 Project Structure

```
signssl-project/
├── README.md                     # This file
├── INSTALLATION.md              # Installation guide
├── TRAINING_GUIDE.md            # Training instructions
├── API_DOCUMENTATION.md         # API reference
├── PROJECT_SUMMARY.md           # Complete overview
├── LICENSE                      # MIT License
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
│
├── configs/
│   └── config.yaml             # Configuration file
│
├── data/
│   ├── raw/                    # Raw data
│   ├── labeled/                # Labeled data
│   ├── unlabeled/              # Unlabeled data
│   ├── processed_labeled.npz   # Processed labeled data
│   ├── processed_unlabeled.npz # Processed unlabeled data
│   └── label_mapping.json      # Label mappings
│
├── src/
│   ├── collect_data.py         # Data collection
│   ├── preprocess.py           # Preprocessing
│   ├── dataset.py              # PyTorch datasets
│   ├── model.py                # Model architecture
│   ├── train.py                # Training script
│   ├── pseudo_label.py         # Pseudo-labeling
│   ├── infer.py                # Real-time inference
│   ├── export.py               # Model export
│   └── utils.py                # Utilities
│
├── models/
│   ├── best_model.pth          # Best trained model
│   ├── quantized_model.pth     # Quantized model
│   ├── model_scripted.pt       # TorchScript export
│   └── model.onnx              # ONNX export
│
├── examples/
│   └── generate_synthetic_data.py  # Generate test data
│
├── create_models.py            # Create initial models
├── setup_and_run.py            # Automated setup
├── quick_start.bat             # Windows quick start
└── test_system.py              # System tests
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
data:
  frames_per_clip: 16           # Number of frames per clip
  frame_height: 224             # Frame height
  frame_width: 224              # Frame width

model:
  encoder_type: 'resnet18'      # 'resnet18' or 'mobilenet_v2'
  hidden_dim: 512               # Hidden dimension
  num_heads: 8                  # Attention heads
  num_transformer_layers: 4     # Transformer layers

training:
  batch_size: 8                 # Batch size
  learning_rate: 0.0001         # Learning rate
  epochs: 50                    # Training epochs
  mixed_precision: true         # Use AMP

semi_supervised:
  pseudo_label_threshold: 0.9   # Confidence threshold
  
inference:
  sliding_window_stride: 4      # Inference stride
  confidence_threshold: 0.7     # Display threshold
```

## 🧪 Testing

Run system tests to verify installation:

```bash
python test_system.py
```

## 📈 Performance

**Expected Results:**
- Training Accuracy: >90% (with sufficient data)
- Validation Accuracy: >80%
- Inference Speed: <50ms per prediction
- Model Size: ~20-50MB (quantized: ~10-20MB)

## 🛠️ Advanced Usage

### Monitor Training with TensorBoard

```bash
tensorboard --logdir logs/train
```

### Use Quantized Model for Faster Inference

```bash
python src/infer.py --model models/quantized_model.pth
```

### Export to ONNX for Deployment

```bash
python src/export.py --formats onnx
```

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce `batch_size` in config.yaml
- Use `mobilenet_v2` encoder
- Disable mixed precision

**Low Accuracy?**
- Collect more training data (50+ samples per sign)
- Increase training epochs
- Use pseudo-labeling

**Slow Inference?**
- Use quantized model
- Reduce `frames_per_clip`
- Increase `sliding_window_stride`

See [INSTALLATION.md](INSTALLATION.md) for more troubleshooting tips.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{signssl2024,
  title={Gloss-Free, Semi-Supervised Sign Language Recognition},
  author={Sign SSL Project},
  year={2024},
  howpublished={\url{https://github.com/yourusername/signssl-project}}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe for landmark extraction
- PyTorch team for the framework
- Sign language research community

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

**Built with ❤️ for accessibility and inclusion**
