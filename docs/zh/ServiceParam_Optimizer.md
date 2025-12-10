# 服务化自动寻优工具
## 简介

**自动寻优工具**：是一个基于PSO粒子寻优算法的服务化参数自动寻优工具，支持对 `MindIE` 和 `VLLM` 进行自动寻优，获取符合时延要求的最佳吞吐参数组合。

**基本概念**：

- `MindIE`、`VLLM`：服务化框架，支持对模型进行服务化部署。
- `Aisbench`、`VLLM_Benchmark`：推理性能评测工具，支持对服务化进行推理性能评测。

**版本配套关系**：

| 版本配套关系 |     CANN     |     框架     |
|:-------------:|:------------:|:--------------:|
|     MindIE当前版本      | CANN 8.1.RC1 | MindIE 2.0.RC1 |
|     VLLM当前版本      | CANN 8.2.RC1 | VLLM 0.8.4 |

## 使用前准备
### 环境准备
1. 准备好能正常运行服务化（如`MindIE Service/VLLM Server`）和测评工具（如`vllm_benchmark/aisbench`）的环境。
2. 安装自动寻优工具，命令如下：
    ```
    如果使用轻量化的方式进行寻优则只需安装最少的依赖即可
    git clone https://gitcode.com/Ascend/msmodeling.git
    cd msmodeling/msModeling/src/optimize
    pip install -e .[real] 

    仿真模式需安装相关加速包
    pip install -e .[speed]  
    如果上述安装失败，可尝试安装较少依赖的三方包，但训练模型时，大数据量时性能较低。
    pip install -e .[train] 

    export PYTHONPATH=$PWD:$PYTHONPATH
    ```

### 约束
- 由于工具涉及使用MindIE镜像，需遵从其启动方式，PD分离场景mindie使用k8s等技术，需用户自行注意相关风险。

## 快速入门
1. 完成[使用前准备](#使用前准备)章节要求；

2. 修改配置文件：启动寻优前需用户按照实际情况配置`config.toml`（msmodeling/msModeling/src/optimize/experimental/config.toml），包括寻优参数、测评工具参数、服务化参数。参考[配置文件说明](#配置文件说明)章节完成配置。

3. 启动寻优：完成上述步骤后，执行以下命令，一键启动轻量化自动寻优：
    ```
    msmodeling optimizer
    ```
    默认执行的是基于`Aisbench`的`MindIE`服务化参数寻优。

4. 查看结果：寻优时间由模型大小和数据集大小决定，一般在4~8小时完成，结束后会在当前路径下输出一个名为 [`data_storage_*.csv`](#输出文件说明) 的文件，其中记录了各组参数的性能。


## 工具原理
工具支持仿真与轻量化两种模式，主要包括三大核心功能模块:

**参数寻优模块**:
利用PSO粒子寻优算法自动生成服务化参数组合，不断逼近最优解；同时，Early Rejection算法通过理论建模、调优经验及部分实测数据对服务化参数完成早期评估；

**仿真模块**:
基于XGBoost模型对大模型推理时长进行精准预测，结合服务化调度的虚拟时间轴技术，加速服务化参数验证速度。

**参数验证模块**: 
自动化启动服务化进程与测评工具进程，进行参数测试，获取性能结果。当前已支持的测评工具包括aisbench, vllm_benchmark。

服务化自动寻优工具能够基于以上功能模块，自动推荐吞吐较优的服务化参数组合，使用时有两种模式：

- [轻量化模式](#轻量化模式) :
注重精度和可靠性，结合参数验证、参数寻优模块，通过真机实测给出可靠的服务化参数推荐值。

- [仿真模式](#仿真模式) :
注重速度及资源占用，调动所有模块快速、精确地预测各组参数的吞吐，在较低NPU资源占用的前提下给出服务化参数推荐值。

目前工具已基于llama3-8b和qwen3-8b通过验证，理论上不限制支持模型范围，后续计划扩大支持范围的验证。


## 轻量化模式
如果需要设置环境变量作用于mindie/vllm服务，只需在运行工具前设置环境变量即可，例如：
```
export ASCEND_RT_VISIBLE_DEVICES=0
```
工具会在寻优过程中自动设置（仿真和轻量化模式均适用）。
前置条件准备就绪后，执行以下命令，一键启动轻量化自动寻优：
```
msmodeling optimizer
```


|参数|说明|是否必选|
|---|---|---|
|-lb, --load_breakpoint|控制是否从断点恢复寻优过程，默认关闭。|否|
|-d, --deploy_policy|设置部署策略，即单机或多机部署，默认为单机部署。若为多机参数需修改为`multiple`|否|
|--backup|决定是否在寻优过程中备份数据，默认False，不开启，用法例如--backup。|否|
|-b, --benchmark_policy|可选：`vllm_benchmark/ais_bench`，默认为'ais_bench'，VLLM框架下需要使用vllm_benchmark/ais_bench.用户需自行选择适配的推理框架以及测试框架|否|
|-e, --engine|指定推理框架，可选：`mindie/vllm`，默认`mindie`。|否|
|--pd|指定推理框架模式pd竞争或pd分离，可选：`competition/disaggregation`，默认`competition`。|否|

以上参数均为可选，若在VLLM场景下使用`vllm_benchmark`测评工具可参考
```
msmodeling optimizer -e vllm -b vllm_benchmark
```

## 仿真模式
### 训练模型
参照[服务化调优工具手册](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/mindieprofiling_0001.html) 开启profiling实际跑一遍MindIE推理服务的测试脚本，将采集的profiling数据进行解析然后用于训练模型。profiling采集数据需要包括batch_type, batch_size, forward_time, batch_end_time(ms), request_recv_token_size, request_reply_token_size, need_blocks, request_execution_time(ms), first_token_latency(ms)。

```
msmodeling train -i=/path/to/input -o=/path/to/output 
```
|参数|说明|是否必选|
|---|---|---|
|-i, --input|输入数据目录，这里所需的数据即profiling的输出路径。|是|
|-o, --output|输出目录，建议输出在modelevalstate下面创建一个/result/latency_model目录来存放。若未指定则会在当前目录下生成。|否|
|-t, --type|框架类型，可选 `mindie/vllm`，默认 `mindie`。|否|

### 启动寻优
- 寻优时需开启环境变量
```
export MODEL_EVAL_STATE_ALL=True。
export MODEL_EVAL_STATE_IS_SLEEP_FLAG=True。
```
```
msmodeling optimizer
```

## 配置文件说明

**寻优参数**： `n_particles` （寻优种子数）、`iters` （迭代轮次数）、 `tpot_slo` （`time_per_output_token`的限制时延）等。
用户可根据预估时间来自行配置种子和迭代次数。我们单个种子使用时间为拉起服务+测试数据。比如用户拉起服务+完成测试需9-10min，且愿意用8小时来进行寻优，则一共可跑约50个种子，建议用户配置5 * 10。设置种子数为10，迭代次数为5，建议用户配置种子数为迭代次数的2倍左右。
|参数|说明|建议值|
|---|---|---|
|n_particles|寻优种子数，即一组生成的参数组合数。| 建议设为 15 ~ 30，范围为1-1000 |
|iters|迭代轮次数。| 建议设为 5 ~ 10，范围为1-1000 |
|ttft_penalty| `time_to_first_token` 即首token时延超时惩罚系数，若对 `time_to_first_token` 没有时延要求设置为0即可。| 按需开启设为1 |
|tpot_penalty|`time_per_output_token` 即非首token时延超时惩罚系数，若对`time_per_output_token`没有时延要求设置为0即可。| 按需开启设为1 |
|success_rate_penalty|请求成功率惩罚系数，默认值为5。| 建议设为5 |
|ttft_slo|`time_to_first_token`的限制时延。| 单位s，如对`time_to_first_token`限制为2s内，则设为2 |
|tpot_slo|`time_per_output_token`的限制时延。| 单位s，如对`time_per_output_token`限制为50ms内，则设为0.05 |
|service|标注多机启动时为主机或从机| 多机场景下从机设为 `slave` |
|sample_size|对原始数据集采样大小，用采样后的数据进行调优，可增加寻优效率| 建议设为原数据集请求的1 / 3, 要求不小于1000 |

**测评工具参数**：
若使用`aisbench`测评，需修改以下参数，可以参照[aisbench 使用说明](https://gitee.com/aisbench/benchmark/blob/master/README.md)进行修改。

|参数|说明|
|---|---|
|models| 指定模型任务，可根据[模型配置说明](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/models.md)进行配置。|
|datasets| 指定数据集任务，可根据[数据集准备指南](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/datasets.md)进行配置 |
|mode| 运行模式。可根据[运行模式说明](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/mode.md)进行配置。| 
|num_prompts| 控制运行数据集的条数，`mode`为`perf`时有效。| 

若使用`vllm_benchmark`测评，需修改以下参数: 

|参数|说明|
|---|---|
|host| 主机ip，需与`[vllm.command]`中的`host`保持一致，可设为`127.0.0.1`。|
|port| 端口号，需与`[vllm.command]`中的`port`保持一致。| 
|model| 模型路径, 需与`[vllm.command]`中的`model`保持一致。| 
|served_model_name| 模型名称, 需与`[vllm.command]`中的`served_model_name`保持一致。|
|dataset_name| 数据集名称。|
|dataset_path| 数据集路径。|
|num_prompts| 控制运行数据集的条数。| 
|others| 拼接其他参数，注意参数间使用空格分隔，参数内部不能留有空格。如`--ignore-eos --custom-output-len 1500`。| 

**服务化参数**: 可以参考[MindIE server 配置参数说明](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html)进行修改。
服务化参数可直接指定参数的范围，如配置服务化参数 `max_batch_size` 的寻优搜索空间为 10 ~ 400，则可设置：
```
[[mindie.target_field]]
"name": "max_batch_size",   // 服务化参数名称
"config_position": "BackendConfig.ScheduleConfig.maxBatchSize", // 服务化参数在MindIE Server中的位置
"min": 10,      // 最小值
"max": 400,     // 最大值
"dtype": "int"  // 数据类型
```
此外，也可设置参数与另一参数相关，如 `max_prefill_batch_size` 与 `max_batch_size` 相关，`max_prefill_batch_size = ratio * max_batch_size (0 < ratio < 1)`则可设置：
```
[[mindie.target_field]]
"name": "max_prefill_batch_size",
"config_position": "BackendConfig.ScheduleConfig.maxPrefillBatchSize",
"min": 0,
"max": 1,
"dtype": "ratio", 
"dtype_param": "max_batch_size" // 表明该参数与max_batch_size相关
```
使用vllm框架时，需修改`config.toml`中的`[vllm.command]`参数，如：
```
[vllm.command]
host = "127.0.0.1"
port = 8000
model = "/workspace/vllm/models/llama-2-7b-chat-hf"
served_model_name = "llama-2-7b-chat-hf"
others = ""
```
|参数|说明|
|---|---|
|host| 主机ip，需与`[vllm_benchmark.command]`中的`host`保持一致，可设为`127.0.0.1`。|
|port| 端口号，需与`[vllm_benchmark.command]`中的`port`保持一致。| 
|model| 模型路径, 需与`[vllm_benchmark.command]`中的`model`保持一致。| 
|served_model_name| 模型名称, 需与`[vllm_benchmark.command]`中的`served_model_name`保持一致。|
|others| 拼接其他参数，注意参数间使用空格分隔，参数内部不能留有空格。如：`--tensor-parallel-size 2 --no-enable-prefix-caching`。|


## pd分离
当前工具只支持mindie的A2单机pd分离，且需要k8s部署。需保证能正常使用k8s拉起mindie服务。
需要在`config.toml`中配置kubectl_default_path字段，将该字段配置为k8s安装脚本解压后的单机执行目录，目录结构需要为：
```
K8s_v1.23_MindCluster.7.1.RC1.B098.aarch/
├── all_label_a2.sh
├── all_label_a3.sh
├── Ascend-docker-runtime_7.1.RC1_linux-aarch64.run
├── Ascend-mindxdl-ascend-operator_7.1.RC1_linux-aarch64/
├── Ascend-mindxdl-clusterd_7.1.RC1_linux-aarch64/
├── Ascend-mindxdl-device-plugin_7.1.RC1_linux-aarch64/
├── Ascend-mindxdl-noded_7.1.RC1_linux-aarch64/
├── Ascend-mindxdl-volcano_7.1.RC1_linux-aarch64/
├── k8s
│   ├── alpine.tar
│   ├── calico3_23.yaml
│   ├── k8s1_23_0+calico3_23.tar.gz
│   └── ubuntu-18.04.tar
└── kubernetes
│   ├── Packages.gz
│   ├── kubeadm_1.23.0-00_arm64.deb
│   ├── kubectl_1.23.0-00_arm64.deb
│   ├── kubelet_1.23.0-00_arm64.deb
│   ├── ...
│   └── zlib1g_1%3a1.2.11.dfsg-2ubuntu9.2_arm64.deb
└── kubernetes_deploy_scripts_latest
    ├──boot_helper
    ├──chat.sh
    ├──conf
    ├──delete.sh
    ├──deploy_ac_job.py
    ├──deployment
    ├──deploy.sh
    ├──envcheck.sh
    ├──gen_ranktable_helper
    ├──log.sh
    ├──pd_scripts_single
    ├──show_logs.sh
    ├──user_config.json
    ├──user_config_base_A3.json
```
即配置
```
kubectl_default_path = "K8s_v1.23_MindCluster.7.1.RC1.B098.aarch/kubernetes_deploy_scripts_latest" #使用绝对路径
```
如果需要配置pd配比的参数寻优只需在config.toml的mindie配置中添加如下参数：
```
[[mindie.target_field]]
name = "default_p_rate"
config_position = "default_p_rate"
min = 1
max = 3
dtype = "int"
value = 1
[[mindie.target_field]]
name = "default_d_rate"
config_position = "default_d_rate"
min = 1
max = 3
dtype = "share"
dtype_param = "default_p_rate"
```
## [插件模式](../../msModeling/src/optimize/experimental/optimizer/plugins/plugin.md)
现在寻优工具支持用户自定义推理框架以及测试工具，用户可以根据自己的需求配置。只需适配我们的插件模式，注册对应的插件即可

## 日志说明
寻优过程中默认日志为INFO级别，如果用户想看每一轮具体的日志，可以在使用工具前设置
```
export MODELEVALSTATE_LEVEL=DEBUG
```
对于每一轮的运行状态会进行输出，我们将具体的MindIE/VLLM日志重定向在/tmp目录下，可以根据打屏信息获取具体文件路径查看MindIE/VLLM运行状态。

# 输出结果
## 输出文件说明
可以使用export MODEL_EVAL_STATE_CUSTOM_OUTPUT来设置输出路径。
输出件为csv，若不指定输出路径则在modelevalstate/result/store下存放。
输出csv中的每一行对应一组参数，前四列为性能指标。用户可以根据需求筛选满足要求的性能行，将MindIE参数以及ais_bench/vllm_benchmark的参数改为csv中的数据即可。
| 列名 | 说明 |
| --- | --- |
| generate_speed | 吞吐。 |
| time_to_first_token | ttft时延，单位为秒。 |
| time_per_output_token | tpot时延，单位为秒。 |
| success_rate | 测试返回请求成功率。 |
| throughput | 测试吞吐，单位为请求数/秒。 |
| CONCURRENCY | 并发数。 |
| REQUESTRATE | 发送速率。 |
| error | 记录这次参数没有正常执行的原因，在发送错误时记录。 |
| backup | 数据记录地址，当开启--backup时记录。 |
| real_evaluation | 标记数据是否由真实测试结果得到。false代表该组数据由gp模型预测得到。 |
| fitness | 寻优算法优化值，该值越小代表该组参数效果越好 |
| num_prompts | 记录这次寻优测评工具发送的请求数。 |

其余列为对应的MindIE或VLLM的config.toml参数。

# FAQ
安装工具时如果安装失败Could not fetch URL https://mirrors.aliyun.com/pypi/simple/poetry 可尝试
```
pip install --trusted-host mirrors.aliyun.com .[speed]
```