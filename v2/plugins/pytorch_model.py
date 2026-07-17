"""
PyTorch Model Plugin
Real AI inference using PyTorch
"""
import sys
sys.path.insert(0, '..')

from core.plugin_base import PluginBase
from typing import Dict, Any
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Simple CNN for demonstration"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PyTorchModelPlugin(PluginBase):
    """
    PyTorch-based AI model plugin.
    Demonstrates real GPU-accelerated inference.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the PyTorch model"""
        try:
            self.model = SimpleCNN()
            
            # Detect best device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded on: {self.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def config(self) -> Dict[str, Any]:
        return {
            'name': 'PyTorchCNN',
            'version': '1.0.0',
            'description': 'PyTorch CNN model for image classification',
            'category': 'ai',
            'author': 'Pluginfer Team',
            'framework': 'pytorch',
            'model_type': 'cnn',
            'input_shape': [1, 28, 28],
            'output_classes': 10,
            'gpu_accelerated': True
        }
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on input tensor.
        
        Expected input:
            {
                'tensor': torch.Tensor or list/array that can be converted
                OR
                'shape': [batch, channels, height, width] for random input
            }
        """
        with torch.no_grad():
            # Get or create input tensor
            if 'tensor' in input_data:
                if isinstance(input_data['tensor'], torch.Tensor):
                    input_tensor = input_data['tensor']
                else:
                    input_tensor = torch.tensor(input_data['tensor'], dtype=torch.float32)
            elif 'shape' in input_data:
                # Generate random input for testing
                input_tensor = torch.randn(*input_data['shape'])
            else:
                # Default: single 28x28 image
                input_tensor = torch.randn(1, 1, 28, 28)
            
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            output = self.model(input_tensor)
            
            # Get predictions
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': probabilities[0].cpu().tolist(),
                'input_shape': list(input_tensor.shape),
                'output_shape': list(output.shape),
                'device': str(self.device)
            }


if __name__ == "__main__":
    # Test the plugin
    try:
        plugin = PyTorchModelPlugin()
        
        print("\nTesting PyTorchModelPlugin...")
        print("\nPlugin Config:", plugin.config())
        
        # Test with random input
        test_input = {'shape': [1, 1, 28, 28]}
        
        print("\n🧪 Running inference...")
        result = plugin.execute(test_input)
        
        print(f"\n✅ Results:")
        print(f"   Predicted Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Device: {result['device']}")
        print(f"   Execution Time: {result['_metadata']['execution_time']:.4f}s")
        
        # Run benchmark
        print("\n📊 Running benchmark (10 iterations)...")
        times = []
        for i in range(10):
            result = plugin.execute(test_input)
            times.append(result['_metadata']['execution_time'])
        
        print(f"   Average: {sum(times)/len(times):.4f}s")
        print(f"   Min: {min(times):.4f}s")
        print(f"   Max: {max(times):.4f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure PyTorch is installed: pip install torch")
