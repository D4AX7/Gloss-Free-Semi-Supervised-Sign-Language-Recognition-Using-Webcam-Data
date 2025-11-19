import torch
import torch.quantization
import yaml
import argparse
from pathlib import Path

from model import create_model
from utils import load_checkpoint


class ModelExporter:
    def __init__(self, config, model_path, device='cpu'):
        self.config = config
        self.device = device
        
        # Load model
        self.model = create_model(config).to(device)
        load_checkpoint(model_path, self.model)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def export_torchscript(self, output_path):
        """Export model to TorchScript"""
        print("Exporting to TorchScript...")
        
        # Create dummy inputs
        batch_size = 1
        frames_per_clip = self.config['data']['frames_per_clip']
        height = self.config['data']['frame_height']
        width = self.config['data']['frame_width']
        
        dummy_frames = torch.randn(batch_size, 3, frames_per_clip, height, width).to(self.device)
        dummy_hand_lms = torch.randn(batch_size, frames_per_clip, 126).to(self.device)
        dummy_pose_lms = torch.randn(batch_size, frames_per_clip, 132).to(self.device)
        
        # Trace model
        traced_model = torch.jit.trace(
            self.model,
            (dummy_frames, dummy_hand_lms, dummy_pose_lms)
        )
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced_model.save(str(output_path))
        
        print(f"TorchScript model saved to {output_path}")
        
        # Verify
        loaded_traced = torch.jit.load(str(output_path))
        with torch.no_grad():
            original_out = self.model(dummy_frames, dummy_hand_lms, dummy_pose_lms)
            traced_out = loaded_traced(dummy_frames, dummy_hand_lms, dummy_pose_lms)
            diff = torch.abs(original_out - traced_out).max().item()
            print(f"Max difference between original and traced: {diff:.6f}")
    
    def export_onnx(self, output_path):
        """Export model to ONNX"""
        print("Exporting to ONNX...")
        
        # Create dummy inputs
        batch_size = 1
        frames_per_clip = self.config['data']['frames_per_clip']
        height = self.config['data']['frame_height']
        width = self.config['data']['frame_width']
        
        dummy_frames = torch.randn(batch_size, 3, frames_per_clip, height, width).to(self.device)
        dummy_hand_lms = torch.randn(batch_size, frames_per_clip, 126).to(self.device)
        dummy_pose_lms = torch.randn(batch_size, frames_per_clip, 132).to(self.device)
        
        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            self.model,
            (dummy_frames, dummy_hand_lms, dummy_pose_lms),
            str(output_path),
            export_params=True,
            opset_version=self.config['export']['onnx_opset_version'],
            do_constant_folding=True,
            input_names=['frames', 'hand_landmarks', 'pose_landmarks'],
            output_names=['logits'],
            dynamic_axes={
                'frames': {0: 'batch_size'},
                'hand_landmarks': {0: 'batch_size'},
                'pose_landmarks': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model saved to {output_path}")
        
        # Verify with onnxruntime
        try:
            import onnxruntime as ort
            
            ort_session = ort.InferenceSession(str(output_path))
            
            # Prepare inputs
            ort_inputs = {
                'frames': dummy_frames.cpu().numpy(),
                'hand_landmarks': dummy_hand_lms.cpu().numpy(),
                'pose_landmarks': dummy_pose_lms.cpu().numpy()
            }
            
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare
            with torch.no_grad():
                pytorch_output = self.model(dummy_frames, dummy_hand_lms, dummy_pose_lms).cpu().numpy()
            
            diff = abs(pytorch_output - ort_outputs[0]).max()
            print(f"Max difference between PyTorch and ONNX: {diff:.6f}")
            
        except ImportError:
            print("onnxruntime not installed, skipping verification")
    
    def export_quantized(self, output_path):
        """Export quantized model"""
        print("Exporting quantized model...")
        
        # Move model to CPU for quantization
        model_cpu = self.model.cpu()
        model_cpu.eval()
        
        # Dynamic quantization (easiest and fastest)
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Save quantized model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'config': self.config
        }, output_path)
        
        print(f"Quantized model saved to {output_path}")
        
        # Compare model sizes
        original_size = sum(p.numel() * p.element_size() for p in model_cpu.parameters()) / (1024 ** 2)
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 ** 2)
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")
        
        # Benchmark inference speed
        batch_size = 1
        frames_per_clip = self.config['data']['frames_per_clip']
        height = self.config['data']['frame_height']
        width = self.config['data']['frame_width']
        
        dummy_frames = torch.randn(batch_size, 3, frames_per_clip, height, width)
        dummy_hand_lms = torch.randn(batch_size, frames_per_clip, 126)
        dummy_pose_lms = torch.randn(batch_size, frames_per_clip, 132)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = quantized_model(dummy_frames, dummy_hand_lms, dummy_pose_lms)
        
        # Benchmark
        import time
        num_runs = 100
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = quantized_model(dummy_frames, dummy_hand_lms, dummy_pose_lms)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        print(f"Average inference time (quantized): {avg_time:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='Export trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Trained model path')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['torchscript', 'onnx', 'quantized'],
                       choices=['torchscript', 'onnx', 'quantized'], help='Export formats')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return
    
    # Create exporter
    exporter = ModelExporter(config, args.model, device=args.device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to requested formats
    if 'torchscript' in args.formats:
        exporter.export_torchscript(output_dir / 'model_scripted.pt')
        print()
    
    if 'onnx' in args.formats:
        exporter.export_onnx(output_dir / 'model.onnx')
        print()
    
    if 'quantized' in args.formats:
        exporter.export_quantized(output_dir / 'quantized_model.pth')
        print()
    
    print("Export complete!")


if __name__ == '__main__':
    main()
