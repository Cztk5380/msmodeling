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

## How to use
We provide a `text_generate.py` command line interface to simulate the text generation. The script supports text generation with a batch of queries with the same input length and optionally same context length. The table summary of op performance breakdown is provided by default. An option is also provided to dump the chrome trace.

Its general usage is shown below:
```text
usage: text_generate.py [-h] [--machine {A2}] --num-queries NUM_QUERIES --input-length INPUT_LENGTH [--context-length CONTEXT_LENGTH]
                        [--compile] [--dump-input-shapes] [--chrome-trace CHROME_TRACE]
                        [--quantize-linear-action {W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC}]
                        model_id

Run a simulated LLM inference pass and dump the perf result.

positional arguments:
  model_id              Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf').

options:
  -h, --help            show this help message and exit
  --machine {A2}        The machine type for simulation. (default: A2)
  --num-queries NUM_QUERIES
                        Number of inference queries to run in a batch. (default: None)
  --input-length INPUT_LENGTH
                        The length of the new input tokens for each query. (default: None)
  --context-length CONTEXT_LENGTH
                        The context length for each query. Defaults to 0. (default: 0)
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --dump-input-shapes   If set, group the table average by input shapes (default: False)
  --chrome-trace CHROME_TRACE
                        Generate chrome trace file (default: None)
  --quantize-linear-action {W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC}
                        Quantize all linear layers in the model from choices (default: None)
```

### Run Prefill
To run a prefill of Qwen3-32B with two requests with 3500-token input length each on A2. You can run the following command:
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 2 --input-length 3500 --machine A2
```
You can also quantize the linear with various quantization schemes, such as W8A8 dynamic quantization and with 4500-token context as the prefix:
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 2 --input-length 3500 --context-length 4500 --machine A2 --quantize-linear-action W8A8_DYNAMIC
```

### Run Decode
Running decode is similar by tweaking the input length and context length. Usually, the input length is 1.
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 10 --input-length 1 --context-length 4500 --machine A2 --quantize-linear-action W8A8_STATIC
```

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