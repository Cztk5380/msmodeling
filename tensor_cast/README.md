## Introduction
TensorCast is a performance simulation and analysis framework for PyTorch programs. It empowers developers and researchers to predict the performance of their neural network models on specific hardware configurations without needing access to the physical machine.

At its core, TensorCast operates as a "virtual machine" or a runtime simulator. Instead of executing computations on a live accelerator (NPU or GPU), it intercepts a PyTorch program's computational graph and simulates its execution on a user-defined MachineConfig. This configuration specifies the target hardware's characteristics, such as theoretical compute power (TFLOPS), memory bandwidths, cache hierarchies, and interconnect speeds. In order to accurately estimate the optimal performance of the model on the given HW, TensorCast provides a model optimization pipeline including automatic model sharding, quantization and FX-graph optimization converting the source program into an optimal one before conducting the analysis.

By running a model on this "virtual" hardware, TensorCast provides detailed performance insights, including:

- Out-of-the-box support for Huggingface transformer models.

- Support hardware accelerators like Ascend NPU and NVidia GPU.

- Operator-level execution time: Estimated using extensible models like analytic roofline model, empirical data, or ML-based predictors.

- Memory footprint: Tracks total and peak memory allocation.

- Computational characteristics: Analyzes FLOPs (Floating Point Operations) and memory access volume for each operator.

- Advanced Scheduling Simulation: Models complex execution patterns like concurrent computations across multiple streams.

The final output includes both comprehensive summary tables and detailed Chrome Trace files, allowing for deep visualization and identification of performance bottlenecks.

## TODO List
- [ ] Qwen3-32B: op perf model, memory allocation, TP, W8A8 (dynamic quant), interconnect modeling
- [ ] Model: Add more model support (make them compilable): kimi-k2, DSv3-671B, Qwen3-235B, GLM-4.5
- [ ] Model: Support model auto sharding (DP/TP/EP/CP/SP)
- [ ] Model: Support model auto quantization (W8A8, W4A8, C8)
- [ ] Compiler: Complete fusion support for Qwen3-32B
- [ ] PerfModel: Implement empirical model. Collect empirical op perf data.
- [ ] PerfModel: Implement analytic model for key PyTorch and Ascend ops.
- [ ] Machine: Add interconnect modeling.
- [ ] Machine: Support H20 modeling.
- [ ] Runtime: Perf text summary.
- [ ] Runtime: Perf chrome trace output.
- [ ] Runtime: Memory consumption estimation for ops.