## Introduction
TensorCast is a performance simulation and analysis framework for PyTorch programs. It empowers developers and researchers to predict the performance of their neural network models on specific hardware configurations without needing access to the physical machine.

At its core, TensorCast operates as a "virtual machine" or a runtime simulator. Instead of executing computations on a live accelerator, it intercepts a PyTorch program's computational graph and simulates its execution on a user-defined MachineConfig. This configuration specifies the target hardware's characteristics, such as theoretical compute power (TFLOPS), memory bandwidths, cache hierarchies, and interconnect speeds. In order to accurately estimate the optimal performance of the model on the given HW, TensorCast provides a model optimization pipeline including automatic model sharding, quantization and FX-graph optimization converting the source program into an optimal one before conducting the analysis.

By running a model on this "virtual" hardware, TensorCast provides detailed performance insights, including:

- Out-of-the-box support for Huggingface transformer models.

- Support various hardware accelerator devices with simple configurations.

- Operator-level execution time: Estimated using extensible models like analytic roofline model, empirical data, or ML-based predictors.

- Memory footprint: Tracks total and peak memory allocation.

- Computational characteristics: Analyzes FLOPs (Floating Point Operations) and memory access volume for each operator.

- Advanced Scheduling Simulation: Models complex execution patterns like concurrent computations across multiple streams.

The final output includes both comprehensive summary tables and detailed Chrome Trace files, allowing for deep visualization and identification of performance bottlenecks.

## Supported Accelerators
We support most of the AI accelerator devices with simple configurations. We have built-in support for Ascend ATLAS-family accelerators in `device.py` and also provide more device examples under `device_profile_examples` that can be copied into `device_profiles` folder for experiments. Note that these are examples for reference only - we do not guarantee their correctness.

### Custom device types
You may also define your own device types in a Python file and drop it under `device_profiles` folder. TensorCast will load them automatically. Refer to `device.py` for examples how to define a new device.

## How to use
### Supported python versions
3.10+

### Install required packages
```bash
git clone https://gitcode.com/Ascend/msit.git -b msserviceprofiler_dev
cd msit/liuren_modeling/tensor_cast
pip install -r requirements.txt
```

### Run text generation with given query length
We provide a `text_generate.py` command line interface to simulate the text generation. The script supports text generation with a batch of queries with the same input length and optionally same context length. The table summary of op performance breakdown is provided by default. An option is also provided to dump the chrome trace.

Its general usage is shown below:
```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE}]
                        --num-queries NUM_QUERIES --query-length QUERY_LENGTH [--context-length CONTEXT_LENGTH] [--compile] [--compile-allow-graph-break]
                        [--dump-input-shapes] [--chrome-trace CHROME_TRACE]
                        [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}] [--quantize-lmhead]        
                        [--mxfp4-group-size MXFP4_GROUP_SIZE] [--quantize-attention-action {DISABLED,INT8}] [--graph-log-url GRAPH_LOG_URL] [--log-level LOG_LEVEL] [--decode]  
                        [--num-mtp-tokens NUM_MTP_TOKENS] [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--disable-repetition]
                        [--reserved-memory-gb RESERVED_MEMORY_GB] [--world-size WORLD_SIZE] [--tp-size TP_SIZE] [--dp-size DP_SIZE] [--mlp-tp-size MLP_TP_SIZE]
                        [--mlp-dp-size MLP_DP_SIZE] [--lmhead-tp-size LMHEAD_TP_SIZE] [--lmhead-dp-size LMHEAD_DP_SIZE] [--o-proj-tp-size O_PROJ_TP_SIZE]
                        [--o-proj-dp-size O_PROJ_DP_SIZE] [--word-embedding-tp] [--ep] [--enable-redundant-experts] [--enable-external-shared-experts]
                        [--host-external-shared-experts] 
                        model_id

Run a simulated LLM inference pass and dump the perf result.
```
Run `python -m tensor_cast.scripts.text_generate --help` for details.
#### External Shared Experts & Redundant Experts Implementation
The following outlines the implementation logic for External Shared Experts and Redundant Experts.

1. Redundant Experts Only:
Each device will host an additional redundant expert.

2. External Shared Experts Only:
Devices are allocated between external shared experts and routing experts at a ratio of 1:`top_k`. Redundant experts are used to pad routing experts if needed.
For example, if `world_size` is 64, `top_k` is 8, and number of routing experts is 256, 8 devices are assigned to host external shared experts.
The remaining 56 devices are used to distribute 256 routing experts. 32 devices host 5 routing experts each. 24 devices host 4 routing experts and 1 redundant expert.

3. Both External Shared Experts & Redundant Experts Enabled: 
The allocation logic is identical to the "External Shared Experts Only" mode, with one addition: If no redundant experts are needed to pad routing experts (i.e., routing experts are evenly distributed across devices), each device hosting routing experts will host an additional redundant expert.

#### Run Prefill
To run a prefill of Qwen3-32B with two requests with 3500-token input length each on A2. You can run the following command:
```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```
You can also quantize the linear with various quantization schemes, such as W8A8 dynamic quantization and with 4500-token context as the prefix:
```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC
```

#### Run Decode
Running decode is similar by tweaking the input length and context length. Usually, the input length is 1.
```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_STATIC
```

### Benchmark the optimal throughput under SLO constraints
Use `scripts/benchmark.py` to search for optimal throughput given models and device lists.
```
usage: benchmark.py [-h] --input-length INPUT_LENGTH --output-length OUTPUT_LENGTH
                    [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE}]
                    [--model-id MODEL_ID] [--num-devices NUM_DEVICES] [-c CONFIG] [--compile] [--compile-allow-graph-break] [--num-mtp-tokens NUM_MTP_TOKENS]
                    [--ttft-limits TTFT_LIMITS [TTFT_LIMITS ...]] [--tpot-limits TPOT_LIMITS [TPOT_LIMITS ...]] [--mode {decode,prefill,both}]
                    [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                    [--mxfp4-group-size MXFP4_GROUP_SIZE] [--quantize-attention-action {DISABLED,INT8}] [--log-level LOG_LEVEL]

Benchmark LLM inference on given devices and models to search for best throughput under given input/output sequence length and SLO limitations
```
Run `python -m tensor_cast.scripts.benchmark --help` for details.

Example:
Below command line benchmarks the optimal throughput for Qwen3-235B-A22B on 16 A2_280T_64G cards under TTFT/TPOT limits and 3.5k+1.5k input/output.
```bash
python -m tensor_cast.scripts.benchmark --model-id Qwen/Qwen3-235B-A22B --device ATLAS_800_A2_280T_64G --num-devices 16 --input-length 3500 --output-length 1500 --num-mtp-tokens 1 --ttft-limits 3 --tpot-limits 0.1 0.05
```
You may provide a YAML configuration file to replace `--model-id Qwen/Qwen3-235B-A22B --device ATLAS_800_A2_280T_64G --num-devices 16` in the
example above. The configuration file can provide a list of devices and models to benchmark at the same time. Below is an example of the file:
```yaml
devices:
  - "ATLAS_800_A2_280T_64G"
  - "H20"

models:
  "Qwen/Qwen3-32B": [1, 2, 4]  # number of devices to test
  "zai-org/GLM-4.5": [4, 8]
```

### Performance analyze for PD aggregation mode
We provide a script `scripts/performance_analyze.py` to analyze the performance of PD aggregation mode.

**Warning**

Please use it on a Linux system, otherwise a RuntimeError may occur when you use --compile.

#### Quick Start
```bash
python -m tensor_cast.scripts.performance_analyze --model-id Qwen/Qwen3-32B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50
```

#### Result Information
The script will output the performance metrics of PD aggregation mode, including throughput, TTFT, TPOT, and concurrency. Like the example below:
```
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration: 
    Model: Qwen/Qwen3-32B
    Devices: 8 TEST_DEVICE
    TTFT Limits: inf
    TPOT Limits: 50.0
  ----------------------------------------------------------------------------
  Overall Best Configuration: 
    Best Throughput: 2013.44
    TTFT: 16805.59
    TPOT: 49.89
  ----------------------------------------------------------------------------
Top 4 Aggregation Configurations: 
+-----+------------+----------+-------+-------------+---------------+-----------+------------+
| Top | Throughput |   TTFT   |  TPOT | concurrency | total_devices |  parallel | batch_size |
+-----+------------+----------+-------+-------------+---------------+-----------+------------+
|  1  |  2013.44   | 16805.59 | 49.89 |     123     |       8       | tp8pp1dp1 |    123     |
|  2  |  1536.53   | 21028.76 | 49.76 |      98     |       8       | tp4pp1dp2 |     49     |
|  3  |   891.36   | 21079.72 | 48.77 |      56     |       8       | tp2pp1dp4 |     14     |
|  4  |   300.11   | 7848.78  | 48.08 |      16     |       8       | tp1pp1dp8 |     2      |
+-----+------------+----------+-------+-------------+---------------+-----------+------------+
********************************************************************************
2025-12-16 03:31:33,042 - msmodeling_logger - INFO - All experiments completed in 18.33 seconds.
```

#### Parameters
```
Common options:
  --input-length INPUT_LENGTH
                        The input length of the prompt. (default: None)
  --output-length OUTPUT_LENGTH
                        The expected output length. (default: None)
  --device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE}     
                        The device type for benchmarking. (default: None)
  --model-id MODEL_ID   Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf'). (default: None)
  --num-devices NUM_DEVICES
                        Number of devices (default: 1)
  --mtp-acceptance-rate MTP_ACCEPTANCE_RATE [MTP_ACCEPTANCE_RATE ...]
                        Acceptance rate list for MTP (default: [0.9, 0.6, 0.4, 0.2])
  --log-level {debug,info,warning,error,critical}
                        Log level to print (default: info)

Model & Quantization Options:
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --compile-allow-graph-break
                        If set, invoke torch.compile() on the model before inference. (default: False)
  --num-mtp-tokens {0,1,2,3,4}
                        Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek (default: 0)
  --quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: W8A8_DYNAMIC)
  --mxfp4-group-size MXFP4_GROUP_SIZE
                        Group size for MXFP4 quantization (default: 32)
  --quantize-attention-action {DISABLED,INT8}
                        Quantize the KV cache with the given action (default: DISABLED)

Service Options:
  --ttft-limits TTFT_LIMITS
                        TTFT constraints under which to search for the best throughput. inf means no constraint. (default: inf)
  --tpot-limits TPOT_LIMITS [TPOT_LIMITS ...]
                        A list of TPOT constraints under which to search for the best throughput. (default: [50.0])
  --backend {MindIE}    Backend name. (default: MindIE)
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        Max prefill tokens (default: 8192)
  -disagg, --disaggregation
                        If set, run in disaggregation mode. (default: False)
```

#### How to calculate the performance metrics
- TTFT:

  We get average `ttft = sum_for_ttft / concurrency`. For sum_for_ttft, we assume the prefill batch size is the max prefill tokens divided by input length.
  So `prefill_batch_size = max_prefill_tokens // input_length`. And request was processed in
  prefill_batch_size steps one by one. We can get the total ttft time as follows:

  `sum_for_ttft = (prefill_latency * prefill_batch_size) * (1 + calc_nums_for_ttft)) * (calc_nums_for_ttft) / 2`

  For example, if we have 12 requests, and max_prefill_tokens is 8192, input_length is 2048,
  then prefill_batch_size is 4. And 12 requests was processed in 3 steps.
  so 
  
  `sum_for_ttft = (prefill_latency * 4 ) * (1 + 3) * 3 / 2`

  `ttft = sum_for_ttft / 12`

- TPOT:

  We don't consider the bubble time in TPOT calculation.

  `tpot = (ttft + decode_latency * output_length) / output_length`

- Output Throughput
  `output_throughput = 1000 * (output_length * concurrency) / (ttft + tpot * output_length)`


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
pip install lintrunner
lintrunner init  # run once
lintrunner -a  # run every time before code check-in: check and apply necessary changes to follow the coding style
```
Fix the remaining lint issues reported by `lintrunner`.

### Unit tests
Make sure unit tests pass by running: `pytest`
