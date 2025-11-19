import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class CNNEncoder(nn.Module):
    """CNN Encoder for extracting spatial features from frames"""
    
    def __init__(self, encoder_type='resnet18', hidden_dim=512, pretrained=True):
        super(CNNEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == 'resnet18':
            # Use ResNet18
            resnet = models.resnet18(pretrained=pretrained)
            # Remove final FC layer
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 512
        elif encoder_type == 'mobilenet_v2':
            # Use MobileNetV2
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.encoder = mobilenet.features
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Projection layer
        self.projection = nn.Linear(self.feature_dim, hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            features: (B, T, hidden_dim)
        """
        B, C, T, H, W = x.shape
        
        # Reshape to process each frame
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Extract features
        features = self.encoder(x)  # (B*T, feature_dim, h, w)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, 1)  # (B*T, feature_dim, 1, 1)
        features = features.view(B * T, -1)  # (B*T, feature_dim)
        
        # Project
        features = self.projection(features)  # (B*T, hidden_dim)
        
        # Reshape back
        features = features.view(B, T, -1)  # (B, T, hidden_dim)
        
        return features


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerTemporalEncoder(nn.Module):
    """Transformer for temporal modeling"""
    
    def __init__(self, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerTemporalEncoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, hidden_dim)
            mask: Optional attention mask
        Returns:
            output: (B, T, hidden_dim)
        """
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return x


class LandmarkEncoder(nn.Module):
    """Encoder for hand and pose landmarks"""
    
    def __init__(self, landmark_dim, hidden_dim, dropout=0.1):
        super(LandmarkEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(landmark_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, landmark_dim)
        Returns:
            output: (B, T, hidden_dim)
        """
        B, T, D = x.shape
        x = x.view(B * T, D)
        x = self.encoder(x)
        x = x.view(B, T, -1)
        return x


class MultimodalFusion(nn.Module):
    """Attention-based fusion of multimodal features"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(MultimodalFusion, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query, key, value):
        """
        Args:
            query: (B, T, hidden_dim)
            key: (B, T, hidden_dim)
            value: (B, T, hidden_dim)
        Returns:
            output: (B, T, hidden_dim)
        """
        # Multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)
        x = self.norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class SignLanguageRecognitionModel(nn.Module):
    """Complete Sign Language Recognition Model"""
    
    def __init__(self, config):
        super(SignLanguageRecognitionModel, self).__init__()
        
        self.config = config
        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_transformer_layers']
        dropout = config['model']['dropout']
        num_classes = config['model']['num_classes']
        encoder_type = config['model']['encoder_type']
        
        # CNN encoder for frames
        self.cnn_encoder = CNNEncoder(encoder_type, hidden_dim, pretrained=True)
        
        # Landmark encoders
        hand_lm_dim = 126  # 2 hands * 21 landmarks * 3 coords
        pose_lm_dim = 132  # 33 landmarks * 4 values
        
        self.hand_encoder = LandmarkEncoder(hand_lm_dim, hidden_dim, dropout)
        self.pose_encoder = LandmarkEncoder(pose_lm_dim, hidden_dim, dropout)
        
        # Transformer temporal encoder
        self.temporal_encoder = TransformerTemporalEncoder(
            hidden_dim, num_heads, num_layers, dropout
        )
        
        # Multimodal fusion
        self.fusion_hand = MultimodalFusion(hidden_dim, num_heads, dropout)
        self.fusion_pose = MultimodalFusion(hidden_dim, num_heads, dropout)
        
        # Aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, frames, hand_landmarks, pose_landmarks):
        """
        Args:
            frames: (B, C, T, H, W)
            hand_landmarks: (B, T, hand_lm_dim)
            pose_landmarks: (B, T, pose_lm_dim)
        Returns:
            logits: (B, num_classes)
        """
        # Extract frame features
        frame_features = self.cnn_encoder(frames)  # (B, T, hidden_dim)
        
        # Extract landmark features
        hand_features = self.hand_encoder(hand_landmarks)  # (B, T, hidden_dim)
        pose_features = self.pose_encoder(pose_landmarks)  # (B, T, hidden_dim)
        
        # Temporal encoding
        frame_features = self.temporal_encoder(frame_features)
        hand_features = self.temporal_encoder(hand_features)
        pose_features = self.temporal_encoder(pose_features)
        
        # Multimodal fusion
        # Fuse hand landmarks with frame features
        fused = self.fusion_hand(frame_features, hand_features, hand_features)
        
        # Fuse pose landmarks
        fused = self.fusion_pose(fused, pose_features, pose_features)
        
        # Temporal pooling
        fused = fused.permute(0, 2, 1)  # (B, hidden_dim, T)
        pooled = self.temporal_pool(fused).squeeze(-1)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits
    
    def extract_features(self, frames, hand_landmarks, pose_landmarks):
        """Extract features without classification (for self-supervised learning)"""
        frame_features = self.cnn_encoder(frames)
        hand_features = self.hand_encoder(hand_landmarks)
        pose_features = self.pose_encoder(pose_landmarks)
        
        frame_features = self.temporal_encoder(frame_features)
        hand_features = self.temporal_encoder(hand_features)
        pose_features = self.temporal_encoder(pose_features)
        
        fused = self.fusion_hand(frame_features, hand_features, hand_features)
        fused = self.fusion_pose(fused, pose_features, pose_features)
        
        fused = fused.permute(0, 2, 1)
        pooled = self.temporal_pool(fused).squeeze(-1)
        
        return pooled


def create_model(config):
    """Factory function to create model"""
    return SignLanguageRecognitionModel(config)


if __name__ == '__main__':
    # Test model
    import yaml
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    
    # Create dummy inputs
    B, C, T, H, W = 2, 3, 16, 224, 224
    frames = torch.randn(B, C, T, H, W)
    hand_landmarks = torch.randn(B, T, 126)
    pose_landmarks = torch.randn(B, T, 132)
    
    # Forward pass
    logits = model(frames, hand_landmarks, pose_landmarks)
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
