MindStudio-Modeling is a performance simulation and analysis framework for neural network inference workloads, consisting of two core components for predicting and optimizing model performance on target hardware:

1.  **TensorCast**
    *   **Core Purpose**: A PyTorch program performance simulator, functioning as a "virtual machine."
    *   **Main Function**: Intercepts a model's PyTorch computational graph and simulates its execution on a user-defined hardware profile (`DeviceProfile`) without requiring physical hardware.
    *   **Supported Tasks**:
        *   **Text Generation**: Simulates Large Language Model (LLM) inference (e.g., Qwen) via `cli.inference.text_generate`.
        *   **Video Generation**: Simulates the forward pass of diffusion transformer models (e.g., Stable Video Diffusion-like architectures) via `cli.inference.video_generate`.
    *   **Output**: Provides operator-level performance breakdown, memory footprint analysis, FLOPs analysis, and can generate Chrome Trace files for visualization.

2.  **ServingCast**
    *   **Core Purpose**: A suite of tools for system-level inference serving simulation and throughput optimization.
    *   **Main Function**:
        *   **Service Simulation**: Driven by `main.py`, it simulates end-to-end serving scenarios with multiple instances and requests based on YAML configuration files, outputting system-level metrics like throughput, latency (TTFT, TPOT).
        *   **Throughput Optimization**: Via `cli.inference.throughput_optimizer.py`, it automatically searches for the optimal model configuration (parallelism strategy, batch size) to maximize token throughput under specified Service Level Objective (SLO) constraints (e.g., limits on TTFT, TPOT).

**Core Value**: It enables developers to predict model performance, identify bottlenecks, and optimize configurations for target hardware without needing access to the physical devices.

<!-- toc -->

- [More About MsModeling](#more-about-msmodeling)
  - [Key Capabilities](#key-capabilities)
  - [Main Components](#main-components)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [License](#license)

<!-- tocstop -->

## Installation
```bash
git clone https://gitcode.com/Ascend/msmodeling.git -b develop
cd msmodeling
pip install -r requirements.txt
```
**Supported Python versions:** 3.10+

> [!Warning]
> If you are using Windows, note that PyTorch 2.10 may not run properly on your system. For a solution, please refer to [this issue](https://github.com/pytorch/pytorch/issues/166628). If you have not yet installed PyTorch, for optimal compatibility, we strongly recommend using version 2.8 or earlier to ensure the program functions correctly.

### Environment Setup
If you are not using the tools within the msmodeling directory, please set the `PYTHONPATH` before running:

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

## Getting Started

For detailed usage, please refer to the two documentation files:
*  [For service simulation and throughput optimization.](./docs/en/serving_cast_instruct.md)
*  [For TensorCast performance simulation framework.](./docs/en/tensor_cast_instruct.md)

## License
msmodeling has a MulanPSL2-style license, as found in the [LICENSE](LICENSE) file.
