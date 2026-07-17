# Pluginfer 🚀

**GPU-Agnostic AI Execution and Licensing Runtime**

Pluginfer is a modular, hardware-agnostic AI inference framework that allows AI models and workloads to run seamlessly across any hardware—CPU, NVIDIA GPU, AMD GPU, Intel GPU, or Apple Silicon—without dependency on CUDA.

## ✨ Features

### Phase 1 (Current Implementation)

- ✅ **GPU-Agnostic Execution**: Automatic hardware detection and optimization
- ✅ **Plugin Architecture**: Modular plugin system for extensibility
- ✅ **Inference Engine**: High-performance execution with monitoring
- ✅ **QAL (Quantum Acceleration Layer)**: Intelligent workload distribution
- ✅ **Licensing System**: Tiered feature access (Free/Pro/Enterprise)
- ✅ **Hardware Detection**: Support for CUDA, MPS, ROCm, DirectML
- ✅ **Batch Processing**: Efficient parallel execution
- ✅ **Usage Tracking**: Quota management and analytics

### Supported Hardware

| Hardware | Status | Backend |
|----------|--------|---------|
| NVIDIA GPU | ✅ Fully Supported | CUDA (PyTorch) |
| Apple Silicon | ✅ Fully Supported | MPS (PyTorch) |
| AMD GPU | ⚠️ Experimental | ROCm (PyTorch) |
| Intel GPU | 🔄 Planned | DirectML/oneAPI |
| CPU | ✅ Fully Supported | Native |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pluginfer.git
cd pluginfer_v2

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch with GPU support
# For NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon:
pip install torch  # MPS support included by default
```

### Basic Usage

```python
from core import PluginRegistry, InferenceEngine

# Initialize
registry = PluginRegistry("plugins")
engine = InferenceEngine()

# Discover plugins
registry.discover_plugins()

# Get a plugin
plugin = registry.get_plugin('TextProcessor')

# Run inference
input_data = {'text': 'Hello World', 'operation': 'uppercase'}
result = engine.run(plugin, input_data)

print(result['result'])  # Output: HELLO WORLD
```

### Command Line Interface

```bash
# Run with automatic checks
python pluginfer.py

# List available plugins
python pluginfer.py --list-plugins

# Run test inference
python pluginfer.py --test

# Show statistics
python pluginfer.py --stats

# Disable licensing (development mode)
python pluginfer.py --no-license
```

## 📦 Plugin Development

Creating a plugin is simple. Here's a minimal example:

```python
from core.plugin_base import PluginBase
from typing import Dict, Any

class MyPlugin(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            'name': 'MyPlugin',
            'version': '1.0.0',
            'description': 'My custom plugin',
            'category': 'custom'
        }
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input
        self.validate_input(input_data, ['data'])
        
        # Process data
        result = process(input_data['data'])
        
        return {'result': result}
```

Save this as `plugins/my_plugin.py` and it will be automatically discovered!

## 🎯 Example Plugins

### 1. Text Processor
- **File**: `plugins/text_processor.py`
- **Operations**: uppercase, lowercase, reverse, word_count
- **Use Case**: Text manipulation and analysis

### 2. Simple AI
- **File**: `plugins/simple_ai.py`
- **Tasks**: classify, predict, embed
- **Use Case**: Demonstration of AI inference patterns

### 3. PyTorch CNN
- **File**: `plugins/pytorch_model.py`
- **Model**: Convolutional Neural Network
- **Use Case**: Real GPU-accelerated image classification

## ⚡ Quantum Acceleration Layer (QAL)

The QAL intelligently distributes workloads across available hardware:

```python
from core import PluginRegistry, QALController

registry = PluginRegistry()
qal = QALController()

# Discover plugins
registry.discover_plugins()
plugin = registry.get_plugin('SimpleAI')

# Create batch of inputs
batch = [{'data': [i]} for i in range(100)]

# Distribute workload
results = qal.distribute_workload(plugin, batch, strategy='auto')
```

### Distribution Strategies

- **auto**: Intelligent distribution based on hardware
- **round_robin**: Rotate across available devices
- **fastest**: Use only the best device

## 🔐 Licensing System

Pluginfer includes a flexible licensing system with three tiers:

### Free Tier
- ✅ CPU execution
- ✅ 100 inferences/day
- ✅ Single plugin
- ❌ No GPU support
- ❌ No QAL

### Pro Tier ($49/month)
- ✅ GPU support (CUDA, MPS, ROCm)
- ✅ Unlimited inferences
- ✅ Up to 10 plugins
- ✅ QAL enabled
- ✅ Batch size: 32

### Enterprise Tier (Custom)
- ✅ All Pro features
- ✅ Unlimited plugins
- ✅ Multi-GPU support
- ✅ Clustering (Phase 2)
- ✅ Priority support
- ✅ Batch size: 128

### Managing Licenses

```python
from core import LicenseValidator, generate_license

# Check current license
validator = LicenseValidator()
validator.print_license_info()

# Generate a license (admin tool)
license_data = generate_license(
    tier='pro',
    key='PRO-XXXX-XXXX-XXXX',
    valid_days=365,
    device_fingerprint='abc123'
)

# Save to license.json
import json
with open('license.json', 'w') as f:
    json.dump(license_data, f, indent=2)
```

## 🏗️ Architecture

```
pluginfer_v2/
├── core/                          # Core framework
│   ├── __init__.py
│   ├── plugin_base.py            # Base plugin class
│   ├── plugin_registry.py        # Plugin discovery
│   ├── inference_engine.py       # Execution engine
│   ├── hardware_detector.py      # Hardware detection
│   ├── qal_controller.py         # Workload distribution
│   └── license_validator.py      # Licensing system
│
├── plugins/                       # Plugin directory
│   ├── text_processor.py
│   ├── simple_ai.py
│   └── pytorch_model.py
│
├── examples/                      # Usage examples
│   ├── example_basic.py
│   └── example_qal_batch.py
│
├── tests/                         # Unit tests
│
├── pluginfer.py                  # Main CLI application
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🧪 Running Examples

```bash
# Example 1: Basic usage
python examples/example_basic.py

# Example 2: QAL batch processing
python examples/example_qal_batch.py

# Test individual plugins
python plugins/text_processor.py
python plugins/simple_ai.py
python plugins/pytorch_model.py
```

## 📊 Performance Monitoring

Pluginfer automatically tracks execution metrics:

```python
# Get engine statistics
engine.print_stats()

# Get QAL performance summary
qal.print_performance_summary()

# Get plugin-specific stats
stats = plugin.get_stats()
```

## 🔬 Hardware Detection

```python
from core import HardwareDetector

detector = HardwareDetector()
detector.print_device_info()

# Get best device
best = detector.get_best_device()
print(f"Using: {best['name']}")

# Get all devices
devices = detector.detect_all_devices()
for device in devices:
    print(f"{device['type']}: {device['name']}")
```

## 🛣️ Roadmap

### Phase 2 (Planned)
- 🔄 Mesh networking for distributed compute
- 🔄 Advanced plugin optimization
- 🔄 Model format converters (ONNX, TFLite)
- 🔄 Web dashboard for monitoring
- 🔄 API server mode

### Phase 3 (Future)
- 🔄 Kubernetes integration
- 🔄 Edge device support
- 🔄 Model marketplace
- 🔄 Revenue sharing for contributors

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## 📝 License

MIT License - see LICENSE file for details

## 🐛 Troubleshooting

### PyTorch not detecting GPU

```bash
# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Plugin not found

```bash
# Check plugin directory
python pluginfer.py --list-plugins

# Verify plugin syntax
python plugins/your_plugin.py
```

### License validation failed

```bash
# Run without licensing (development)
python pluginfer.py --no-license

# Check license file
cat license.json
```

## 📧 Support

- **Documentation**: See `/docs` folder
- **Issues**: GitHub Issues
- **Email**: support@pluginfer.ai

## 🌟 Acknowledgments

Built with:
- PyTorch for AI/ML
- Python 3.8+
- Love for open-source ❤️

---

**Pluginfer** - Making AI inference truly hardware-agnostic 🚀
