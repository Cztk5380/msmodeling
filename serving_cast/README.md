# How to use

## Install requirements

```bash
git clone https://gitcode.com/Ascend/msmodeling.git -b develop
cd msmodeling
pip install -r ./serving_cast/requirements.txt
```

## Set environment variable
```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

## Run simulation

Its general usage is shown below:
```text
usage: main.py [-h] --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...] --common_config_path COMMON_CONFIG_PATH

Run a service inference simulation driven by JSON configuration files.

required arguments:
  --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...]
                        Path to a YAML file that declares one or more instance groups.
                        Each group defines a homogeneous pool of nodes (role, count, TP/DP parallelism) 
                        and can be mixed-and-matched in a single benchmark run.
  --common_config_path COMMON_CONFIG_PATH
                        Path to a YAML file with global settings: model architecture,
                        request-generation workload, and serving limits.

optional arguments:
  -h, --help            show this help message and exit
  --enable_profiling    Enable profiling during simulation (default: False)​
  --profiling_output_path PROFILING_OUTPUT_PATH​
                        Path to directory where profiling results will be saved (default: ./profiling_results)
```

example:

- basic usage
```bash
python main.py --instance_config_path=./example/instances.yaml --common_config_path=./example/common.yaml 
```

- enable profiling
```bash
python main.py --instance_config_path=./example/instances.yaml --common_config_path=./example/common.yaml --enable_profiling​ 
```

- enable profiling with custom output path
```bash
python main.py --instance_config_path=./example/instances.yaml --common_config_path=./example/common.yaml --enable_profiling --profiling_output_path=/path/to/custom/profiling_dir
```

### Result

After the simulation finishes, a performance summary is printed to the console like following:

```text
         E2E_TIME(s)  TTFT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE     1052.591    0.378    0.301        1500.0         3500.0                           3.327
MIN         1050.000    0.300    0.300        1500.0         3500.0                           2.978
MAX         1175.500    0.600    0.336        1500.0         3500.0                           3.334
MEDIAN      1050.100    0.400    0.300        1500.0         3500.0                           3.334
P75         1050.125    0.400    0.300        1500.0         3500.0                           3.334
P90         1050.200    0.500    0.300        1500.0         3500.0                           3.334
P99         1175.500    0.600    0.336        1500.0         3500.0                           3.334
======== Overall Summary ========
benchmark_duration(s)          1225.500
total_requests                 100.000
request_throughput(req/s)      0.082
total_input_tokens             150000.000
input_token_throughput(tok/s)  122.399
total_output_tokens            350000.000
output_token_throughput(tok/s) 285.598
```

Metric descriptions:
- E2E_TIME: End-to-end latency per request (issue → last token)
- TTFT: Time-to-first-token
- TPOT: Time-per-output-token after the first token
- OUTPUT_TOKEN_THROUGHPUT: Per-request output-token rate
- request_throughput: System-wide request rate
- input_token_throughput / output_token_throughput: Aggregate token throughput

### Profiling

Profiling is supported in the simulation. You can get more specific information about the performance of the system by viewing the profiling result.

Use the following command to enable profiling:

- enable profiling
```bash
python main.py --instance_config_path=./example/instances.yaml --common_config_path=./example/common.yaml --enable_profiling​ 
```

- enable profiling with custom output path
```bash
python main.py --instance_config_path=./example/instances.yaml --common_config_path=./example/common.yaml --enable_profiling --profiling_output_path=/path/to/custom/profiling_dir
```


The original collected profiling result is stored in the directory ```profiling_output_path/{$time_stamp}```.
The parsed profiling result is stored in the directory ```profiling_output_path/{$time_stamp}_parsed_result```.

A ```chrome_tracing.json``` and a ```profiler.db``` will be generated in parsed_result directory, you can view it by ```chrome://tracing``` or MindStudio Insight


## Performance analyze for PD aggregation mode
We provide a script `scripts/performance_analyze.py` to analyze the performance of PD aggregation mode.

### Quick Start
```bash
cd /path/to/msmodeling
pip install -r ./tensor_cast/requirements.txt
python -m serving_cast.scripts.performance_analyze --model-id Qwen/Qwen3-32B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50
```

### Result Information
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

### Parameters
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

### How to calculate the performance metrics
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