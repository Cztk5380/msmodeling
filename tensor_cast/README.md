## Introduction
TensorCast is a performance simulation and analysis framework for PyTorch programs. It empowers developers and researchers to predict the performance of their neural network models on specific hardware configurations without needing access to the physical machine.

At its core, TensorCast operates as a "virtual machine" or a runtime simulator. Instead of executing computations on a live accelerator, it intercepts a PyTorch program's computational graph and simulates its execution on a user-defined MachineConfig. This configuration specifies the target hardware's characteristics, such as theoretical compute power (TFLOPS), memory bandwidths, cache hierarchies, and interconnect speeds. In order to accurately estimate the optimal performance of the model on the given HW, TensorCast provides a model optimization pipeline including automatic model sharding, quantization and FX-graph optimization converting the source program into an optimal one before conducting the analysis.

By running a model on this "virtual" hardware, TensorCast provides detailed performance insights, including:

- Out-of-the-box support for Huggingface transformer models.

- Support hardware accelerators like Ascend NPU and NVidia GPU.

- Operator-level execution time: Estimated using extensible models like analytic roofline model, empirical data, or ML-based predictors.

- Memory footprint: Tracks total and peak memory allocation.

- Computational characteristics: Analyzes FLOPs (Floating Point Operations) and memory access volume for each operator.

- Advanced Scheduling Simulation: Models complex execution patterns like concurrent computations across multiple streams.

The final output includes both comprehensive summary tables and detailed Chrome Trace files, allowing for deep visualization and identification of performance bottlenecks.

## Supported Accelerators
We support most of the AI accelerator devices from HW vendors: Huawei, NVidia, Alibaba, Baidu, Cambricon, Metax etc. Check `device.py` for details.

### Custom device types
You may also define your own device types in a Python file and drop it under `device_profiles` folder. TensorCast will load them automatically. Refer to `device.py` for examples how to define a new device.

## How to use
### Install required packages
`pip install -r requirements.txt`

### Run text generation with given query length
We provide a `text_generate.py` command line interface to simulate the text generation. The script supports text generation with a batch of queries with the same input length and optionally same context length. The table summary of op performance breakdown is provided by default. An option is also provided to dump the chrome trace.

Its general usage is shown below:
```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,B30A,H20,H100_SXM,H200_SXM,H800_SXM,L20,RTX_PRO_6000D,RTX_6000D,RTX_5090D,RTX_5090Dv2,RTX_4090,RTX_4090D,MLU690,MLU590,MLU580,P800,PPU,C550,ATLAS_850_DV100_425T_84G,ATLAS_850_DV100_425T_84G_PCIE,ATLAS_850_DV100_425T_128G_PCIE,ATLAS_850_DV120_486T_96G_POD,ATLAS_850_DV120_547T_144G_POD,ATLAS_850_DV120_486T_144G,ATLAS_850_DV120_425T_96G,ATLAS_850_DV120_425T_84G}]
                        --num-queries NUM_QUERIES --query-length QUERY_LENGTH [--context-length CONTEXT_LENGTH] [--compile] [--compile-allow-graph-break] [--dump-input-shapes] [--chrome-trace CHROME_TRACE]
                        [--quantize-linear-action {W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC}] [--quantize-lmhead] [--graph-log-url GRAPH_LOG_URL] [--log-level LOG_LEVEL] [--decode] [--num-mtp-tokens NUM_MTP_TOKENS]
                        [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--enable-repetition] [--reserved-memory-gb RESERVED_MEMORY_GB] [--world-size WORLD_SIZE] [--tp-size TP_SIZE] [--dp-size DP_SIZE] [--mlp-tp-size MLP_TP_SIZE] [--mlp-dp-size MLP_DP_SIZE] [--lmhead-tp-size LMHEAD_TP_SIZE]
                        [--lmhead-dp-size LMHEAD_DP_SIZE] [--ep]
                        model_id

Run a simulated LLM inference pass and dump the perf result.

positional arguments:
  model_id              Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf').

options:
  -h, --help            show this help message and exit
  --device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,B30A,H20,H100_SXM,H200_SXM,H800_SXM,L20,RTX_PRO_6000D,RTX_6000D,RTX_5090D,RTX_5090Dv2,RTX_4090,RTX_4090D,MLU690,MLU590,MLU580,P800,PPU,C550,ATLAS_850_DV100_425T_84G,ATLAS_850_DV100_425T_84G_PCIE,ATLAS_850_DV100_425T_128G_PCIE,ATLAS_850_DV120_486T_96G_POD,ATLAS_850_DV120_547T_144G_POD,ATLAS_850_DV120_486T_144G,ATLAS_850_DV120_425T_96G,ATLAS_850_DV120_425T_84G}
                        The device type for simulation. (default: TEST_DEVICE)
  --num-queries NUM_QUERIES
                        Number of inference queries to run in a batch. (default: None)
  --query-length QUERY_LENGTH
                        The length of the new input tokens for each query. (default: None)
  --context-length CONTEXT_LENGTH
                        The context length for each query. Defaults to 0. (default: 0)
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --compile-allow-graph-break
                        If set, invoke torch.compile() on the model before inference. (default: False)
  --dump-input-shapes   If set, group the table average by input shapes (default: False)
  --chrome-trace CHROME_TRACE
                        Generate chrome trace file (default: None)
  --quantize-linear-action {W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: None)
  --quantize-lmhead     Whether to quantize LM Head, off by default since quantizing LM Head usually impact accuracy a lot (default: False)
  --graph-log-url GRAPH_LOG_URL
                        For debug: the path for dumping the compiled graphs if compile is on (default: None)
  --log-level LOG_LEVEL
                        Logging level (default: None)
  --decode              Whether we are doing decode (default: False)
  --num-mtp-tokens NUM_MTP_TOKENS
                        Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek (default: 0)
  --num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE
                        Override the number of hidden layers, for debugging only (default: 0)
  --enable-repetition   Leverage the repetition pattern of the transformer models to save runtime cost (default: False)
  --reserved-memory-gb RESERVED_MEMORY_GB
                        Size of reserved device memory (in GB) that we cannot use from applications. (default: 0)
  --world-size WORLD_SIZE
                        The total number of processes (default: 1)
  --tp-size TP_SIZE     The tp size for the whole model (default: 1)
  --dp-size DP_SIZE     The dp size for the whole model (default: None)
  --mlp-tp-size MLP_TP_SIZE
                        The tp size fo mlp layer, can override tp-size for mlp layer (default: None)
  --mlp-dp-size MLP_DP_SIZE
                        The dp size fo mlp layer, can override dp-size for mlp layer (default: None)
  --lmhead-tp-size LMHEAD_TP_SIZE
                        The tp size fo lm head, can override tp-size for lm head (default: None)
  --lmhead-dp-size LMHEAD_DP_SIZE
                        The dp size fo lm head, can override dp-size for lm head (default: None)
  --ep                  Whether or not to implement expert parallel (default: False)
```

#### Run Prefill
To run a prefill of Qwen3-32B with two requests with 3500-token input length each on A2. You can run the following command:
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```
You can also quantize the linear with various quantization schemes, such as W8A8 dynamic quantization and with 4500-token context as the prefix:
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC
```

#### Run Decode
Running decode is similar by tweaking the input length and context length. Usually, the input length is 1.
```bash
python -m tensor_cast.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_STATIC
```

### Benchmark the optimal throughput under SLO constraints
Use `scripts/benchmark.py` to search for optimal throughput given models and device lists.
```
usage: benchmark.py [-h] --input-length INPUT_LENGTH --output-length OUTPUT_LENGTH
                    [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,B30A,H20,H100_SXM,H200_SXM,H800_SXM,L20,RTX_PRO_6000D,RTX_6000D,RTX_5090D,RTX_5090Dv2,RTX_4090,RTX_4090D,MLU690,MLU590,MLU580,P800,PPU,C550,ATLAS_850_DV100_425T_84G,ATLAS_850_DV100_425T_84G_PCIE,ATLAS_850_DV100_425T_128G_PCIE,ATLAS_850_DV120_486T_96G_POD,ATLAS_850_DV120_547T_144G_POD,ATLAS_850_DV120_486T_144G,ATLAS_850_DV120_425T_96G,ATLAS_850_DV120_425T_84G}]
                    [--model-id MODEL_ID] [--num-devices NUM_DEVICES] [-c CONFIG] [--compile] [--num-mtp-tokens NUM_MTP_TOKENS] [--ttft-limits TTFT_LIMITS [TTFT_LIMITS ...]] [--tpot-limits TPOT_LIMITS [TPOT_LIMITS ...]] [--mode {decode,prefill,both}] [--quantize-linear {W8A8,W4A8}] [--log-level LOG_LEVEL]

Benchmark LLM inference on given devices and models to search for best throughput under given input/output sequence length and SLO limitations

options:
  -h, --help            show this help message and exit
  --input-length INPUT_LENGTH
                        The input length of the prompt. (default: None)
  --output-length OUTPUT_LENGTH
                        The expected output length. (default: None)
  --device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,B30A,H20,H100_SXM,H200_SXM,H800_SXM,L20,RTX_PRO_6000D,RTX_6000D,RTX_5090D,RTX_5090Dv2,RTX_4090,RTX_4090D,MLU690,MLU590,MLU580,P800,PPU,C550,ATLAS_850_DV100_425T_84G,ATLAS_850_DV100_425T_84G_PCIE,ATLAS_850_DV100_425T_128G_PCIE,ATLAS_850_DV120_486T_96G_POD,ATLAS_850_DV120_547T_144G_POD,ATLAS_850_DV120_486T_144G,ATLAS_850_DV120_425T_96G,ATLAS_850_DV120_425T_84G}
                        The device type for benchmarking. (default: )
  --model-id MODEL_ID   Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf'). (default: )
  --num-devices NUM_DEVICES
                        Number of devices (default: 1)
  -c CONFIG, --config CONFIG
                        Configuration file for device list, model list and number of devices, overriding --device, --model-id, --num-devices Example config.yaml format: --------------------------- devices: - "ATLAS_800.A2_280T_64G" - "NVIDIA.H20" models: "Qwen/Qwen3-32B": [1, 2, 4] # number of devices to
                        test "zai-org/GLM-4.5": [4, 8] (default: )
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --num-mtp-tokens NUM_MTP_TOKENS
                        Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek (default: 0)
  --ttft-limits TTFT_LIMITS [TTFT_LIMITS ...]
                        A list of TTFT constraints under which to search for the best throughput. (default: [1])
  --tpot-limits TPOT_LIMITS [TPOT_LIMITS ...]
                        A list of TPOT constraints under which to search for the best throughput. (default: [0.05])
  --mode {decode,prefill,both}
                        Inference mode (default: both)
  --quantize-linear {W8A8,W4A8}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: W8A8)
  --log-level LOG_LEVEL
                        Logging level (default: None)
```
Example:
Below command line benchmarks the optimal throughput for Qwen3-235B-A22B on 16 A2_280T_64G cards under TTFT/TPOT limits and 3.5k+1.5k input/output.
```bash
python -m tensor_cast.scripts.benchmark --model-id Qwen/Qwen3-235B-A22B --device ATLAS_800_A2_280T_64G --num-devices 16 --input-length 3500 --output-length 1500 --num-mtp-tokens 1 --ttft-limits 3 --tpot-limits 0.1 0.05
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

## Contributions
### Installation
`pip install -r requirements.txt`

### Coding style
Use `lintrunner` to make sure the coding style aligns:
```bash
lintrunner init  # run once
lintrunner -a  # run every time before code check-in: check and apply necessary changes to follow the coding style
```
Fix the remaining lint issues reported by `lintrunner`.

### Unit tests
Make sure unit tests pass by running: `pytest`
