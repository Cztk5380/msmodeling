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
usage: text_generate.py [-h] [--device {TEST_DEVICE,B30A,H20,H100_SXM,H200_SXM,H800_SXM,L20,RTX_6000D,RTX_5090D,RTX_4090,RTX_4090D,MLU590,P800,PPU,C550}] --num-queries NUM_QUERIES --input-length INPUT_LENGTH [--context-length CONTEXT_LENGTH]
                        [--max-context-length MAX_CONTEXT_LENGTH] [--compile] [--compile-allow-graph-break] [--dump-input-shapes] [--chrome-trace CHROME_TRACE]
                        [--quantize-linear-action {W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC}] [--graph-log-url GRAPH_LOG_URL] [--log-level LOG_LEVEL] [--decode] [--num-mtp-layers NUM_MTP_LAYERS]
                        [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--enable-repetition]
                        model_id

Run a simulated LLM inference pass and dump the perf result.

positional arguments:
  model_id              Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf').

options:
  -h, --help            show this help message and exit
  --device {TEST_DEVICE,B30A,H20,H100_SXM,H200_SXM,H800_SXM,L20,RTX_6000D,RTX_5090D,RTX_4090,RTX_4090D,MLU590,P800,PPU,C550}
                        The device type for simulation. (default: TEST_DEVICE)
  --num-queries NUM_QUERIES
                        Number of inference queries to run in a batch. (default: None)
  --input-length INPUT_LENGTH
                        The length of the new input tokens for each query. (default: None)
  --context-length CONTEXT_LENGTH
                        The context length for each query. Defaults to 0. (default: 0)
  --max-context-length MAX_CONTEXT_LENGTH
                        Max supported context length for each query. (default: 131072)
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --compile-allow-graph-break
                        If set, invoke torch.compile() on the model before inference. (default: False)
  --dump-input-shapes   If set, group the table average by input shapes (default: False)
  --chrome-trace CHROME_TRACE
                        Generate chrome trace file (default: None)
  --quantize-linear-action {W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: None)
  --graph-log-url GRAPH_LOG_URL
                        For debug: the path for dumping the compiled graphs if compile is on (default: None)
  --log-level LOG_LEVEL
                        Logging level (default: None)
  --decode              Whether we are doing decode (default: False)
  --num-mtp-layers NUM_MTP_LAYERS
                        Number of MTP layers, 0 means disabled - only support models having MTP like DeepSeek (default: 0)
  --num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE
                        Override the number of hidden layers, for debugging only (default: 0)
  --enable-repetition   Leverage the repetition pattern of the transformer models to save runtime cost (default: False)
```

### Run Prefill
To run a prefill of Qwen3-32B with two requests with 3500-token input length each on A2. You can run the following command:
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 2 --input-length 3500 --device TEST_DEVICE
```
You can also quantize the linear with various quantization schemes, such as W8A8 dynamic quantization and with 4500-token context as the prefix:
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 2 --input-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC
```

### Run Decode
Running decode is similar by tweaking the input length and context length. Usually, the input length is 1.
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 10 --input-length 1 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_STATIC
```

## TODO List
- [X] Qwen3-32B: op perf model, memory allocation, TP, W8A8 (dynamic quant), interconnect modeling
- [X] Model: Add more model support (make them compilable): kimi-k2, DSv3-671B, Qwen3-235B, GLM-4.5
- [ ] Model: Support model auto sharding (DP/TP/EP/CP/SP)
  - [X] DP
  - [X] TP
  - [X] EP
  - [ ] CP
  - [ ] SP
- [ ] Model: Support model auto quantization (W8A8, W4A8, C8 etc.)
  - [X] W8A8
  - [X] W4A8
  - [ ] FP8
  - [ ] FP4
  - [ ] C8
- [ ] Compiler: Complete fusion support for models
  - [ ] Qwen3 Dense
  - [ ] DeepSeek
- [ ] PerfModel: Implement empirical model. Collect empirical op perf data.
- [ ] PerfModel: Implement analytic model for key PyTorch and Ascend ops.
- [X] Device: Add interconnect modeling.
- [X] Device: Support H20 modeling.
- [X] Runtime: Perf text summary.
- [X] Runtime: Perf chrome trace output.
- [X] Runtime: Memory consumption estimation for ops.