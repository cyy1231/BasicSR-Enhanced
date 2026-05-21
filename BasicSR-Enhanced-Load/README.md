# BasicSR-Enhanced

<p align="center">
  An enhanced training framework <a href="https://github.com/XPixelGroup/BasicSR">based on BasicSR</a>
</p>

<p align="center">
  Introducing <strong>Gradient Accumulation, Feature Visualization, Weight/Gradient Monitoring,
  and Automated Checkpoint Management</strong>. Additionally, <strong>Network Architecture Export</strong>
  and <strong>Complexity Analysis</strong> utilities are integrated into <code>BaseModel</code>.
</p>

---

## ✨ Overview of New Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Gradient Accumulation** | Accumulate gradients over multiple forwards before a single optimizer step | Train with large effective batch size under limited GPU memory |
| **Feature Visualization** | Capture and save intermediate layer feature maps as normalized image grids | Analyze network behavior, generate figures for papers |
| **Weight/Gradient Monitoring** | Log weight and gradient distributions to TensorBoard histograms | Diagnose vanishing/exploding gradients, monitor training stability |
| **Checkpoint Manager** | Automatically retain only the most recent N checkpoints | Prevent disk space exhaustion from model files |

---

## 🛠️ New Utility Functions in BaseModel

Two helper functions are added to `BaseModel` and invoked automatically before training to generate model structure reports:

### `save_network_architecture(net, save_name='network_arch.txt')`
Exports a layer-by-layer network architecture text file, including:
- Layer name, module type, parameter count, and trainable flag
- Summary of total / trainable / non-trainable parameters

**Example output**:
```
Layer                                         Type                      Params  Trainable
------------------------------------------------------------------------------------------
conv1                                         Conv2d                    9,408   True
body.0                                        ResidualBlock             36,864  True
upsampler                                     PixelShuffle              --      --
------------------------------------------------------------------------------------------
Total parameters:            1,234,567
Trainable parameters:        1,234,567
Non-trainable params:              0
```

### `save_network_complexity(net, save_name='network_complexity.txt')`
Automatically infers input resolution from `opt['datasets']['train']`, then uses `ptflops` to compute and save network complexity:
- Total parameter count
- MACs / FLOPs (if `ptflops` is installed)
- Inferred input resolution

**Example output**:
```
Network Class: CATANet
Input Resolution (LR): 64x64
------------------------------------------------------------
Total parameters:      1,234,567
MACs (GMac):           12.34 GMac
Params (ptflops):      1.23 M
```

> Both files are saved **automatically** to the experiment root directory during training initialization. No manual call is required.

---

## 🔧 Gradient Accumulation

Simulate large-batch training by accumulating gradients over multiple forward-backward passes.

**Key capabilities**:
- **Auto-scaling**: Automatically scales iteration-based configs (`total_iter`, `milestones`, `val_freq`, `print_freq`, `save_checkpoint_freq`) by `accumulation_steps`, ensuring the number of parameter updates remains consistent with the original configuration.
- **Forced flush**: Calls `flush_gradients()` at the end of each epoch to prevent leftover accumulated gradients from being ignored.
- **Distributed support**: Compatible with `DistributedDataParallel`. Loss all-reduce occurs only on the last accumulation step.

### Configuration
```yaml
train:
  accumulation_steps: 4  # 4 forwards = 1 parameter update
  use_accumulation: true
```

### Code Interface (inside model)
```python
# Already encapsulated in BaseModel; model subclasses need no changes
self._init_gradient_accumulation()      # initialization
self._accumulation_step_begin()         # call at start of optimize_parameters
self._scale_loss_for_accumulation(loss)  # scale loss before backward
self._accumulation_step_end(loss_dict)  # call at end; handles step, EMA, and logging
```

---

## 🎯 Feature Visualization

Register `forward_hook` to capture intermediate layer outputs and save them as normalized image grids.

**Key capabilities**:
- **Multi-stage support**: Independent switches for `val` and `test` phases.
- **Quantity limit**: `visualize_max_val_images` restricts how many validation images trigger feature capture, preventing disk bloat.
- **Auto cleanup**: Hooks are removed via `remove_feature_hooks()` with `try-finally` guards to prevent handle leaks.

**Visualization examples**

<p align="center">
  <img src="display\iter_64_CATANet_first_conv.png" width="600">
  <br>
  <em>Feature maps of intermediate layer first_conv (CATANet iteration 64)</em>
</p>

### Configuration
```yaml
network_g:
  visual_config:
    name: YourNetName
    visualize_layers: ['body.0', 'body.2', 'upsampler']  # layer names from named_modules
    visualize_during_val: true
    visualize_during_test: false
    visualize_max_val_images: 3
```

---

## 📊 Weight & Gradient Distribution Logging

Capture gradients of specified layers via `register_full_backward_hook`, and log both weights and gradients as TensorBoard histograms.

**Key capabilities**:
- **Substring matching**: `visualize_weight_layers` supports partial name matching; full paths are not required.
- **Frequency alignment**: Logs by "parameter update step" rather than "forward count", automatically aligning with gradient accumulation.
- **Memory safety**: All early-return branches execute `grads.clear()` to prevent gradient tensors from lingering in GPU memory.

**Visualization examples**

<p align="center">
  <img src="display\weight_and_gradient_visual_sample.png" width="600">
  <br>
  <em>Weight and Gradient maps of intermediate layer...</em>
</p>

### Configuration
```yaml
network_g:
  visual_config:
    name: CATANet
    visualize_weight_layers: ['body.0', 'body.2']  # substring matching
    visualize_weight_freq: 500  # log every 500 parameter updates

logger:
  use_tb_logger: true
```

### TensorBoard Tag Format
```
train/CATANet/weight/body/0
train/CATANet/grad/body/0
```

---

## 💾 Automated Checkpoint Management

Automatically purge outdated model files (`.pth`) and training state files (`.state`), retaining only the most recent N checkpoints.

### Configuration
```yaml
train:
  max_checkpoints: 5
```



## Setup
```bash
set PYTHONPATH=%cd%   /  export PYTHONPATH=$(pwd)
set http_proxy=http://127.0.0.1:7897
```

## Installation

### Step 1: Install PyTorch (Select for your GPU)

> **Note:** This project requires **PyTorch >= 2.0**. Install the wheel that matches your **CUDA version** and **GPU architecture**. Do **not** install PyTorch from `requirements.txt`.

**NVIDIA RTX 50-series (Blackwell, CUDA 12.9):**
```bash
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
```

**NVIDIA RTX 40-series (Ada, CUDA 12.4):**
```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

**NVIDIA RTX 30-series (Ampere, CUDA 11.8):**
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

**CPU-only / macOS:**
```bash
pip install torch torchvision torchaudio
```

### Step 2: Clone and Install Dependencies

```bash
git clone https://github.com/yourname/basicsr-enhanced.git
cd basicsr-enhanced

# Install runtime dependencies (PyTorch excluded)
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

---
## Training

### Quick Start (Single GPU)

```bash
python basicsr/train.py -opt options
```

### Multi-GPU Training (Distributed Data Parallel)

```bash
# 2 GPUs
python -m torchrun --nproc_per_node=2 --master_port=29500 \
    basicsr/train.py -opt options

# 4 GPUs
python -m torchrun --nproc_per_node=4 --master_port=29500 \
    basicsr/train.py -opt options
```

### Resume from Checkpoint

```bash
# Resume from the latest saved state automatically
python basicsr/train.py -opt options --auto_resume

# Or specify a state file manually
python basicsr/train.py -opt options \
    --resume states/500000.state
```


## 📝 Citation

Please also cite the original BasicSR repository:

```bibtex
@article{wang2023basicsr,
  title={BasicSR: Open Source Image and Video Restoration Toolbox},
  author={Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  journal={https://github.com/XPixelGroup/BasicSR},
  year={2023}
}
```

---

## 📜 License

This project is released under the [Apache License 2.0](LICENSE).
Original BasicSR copyright belongs to the XPixel Group.