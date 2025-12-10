# 项目目录

```
msmodeling/                          # 项目根目录
├── README.md                        # 项目说明文档
├── docs/                            # 文档目录
│   ├── images/                      # 图片资源
│   │   └── vulnerability_handling_process.png
│   └── zh/                          # 中文文档
│       ├── mindstudio_vulnerability_handling_procedure.md
│       ├── dir_structure.md         # 目录结构说明（当前文件）
│       ├── release_notes.md         # 发布说明
│       ├── security_statement.md    # 安全声明
│       ├── public_ip_address.md     # 公网地址信息
│       └── ServiceParam_Optimizer.md # 自动寻优工具说明
└── msModeling/                      # 主要代码目录
    └── src/                         # 源代码目录
        └── optimize/                # 自动寻优相关模块
            ├── experimental/        # 实验性功能
            │   ├── __init__.py      # Python包初始化文件
            │   ├── __main__.py      # 程序入口文件
            │   ├── analysis.py      # 分析功能模块
            │   ├── common.py        # 公共功能模块
            │   ├── config/          # 配置相关
            │   │   ├── __init__.py
            │   │   ├── base_config.py
            │   │   ├── config.py
            │   │   ├── custom_command.py
            │   │   └── model_config.py
            │   ├── config.toml      # 配置文件
            │   ├── data_feature/    # 数据特征处理
            │   │   ├── __init__.py
            │   │   ├── dataset.py
            │   │   ├── dataset_with_modin.py
            │   │   ├── dataset_with_swifter.py
            │   │   └── v1.py
            │   ├── inference/       # 推理相关
            │   │   ├── __init__.py
            │   │   ├── common.py
            │   │   ├── constant.py
            │   │   ├── data_format_v1.py
            │   │   ├── dataset.py
            │   │   ├── file_reader.py
            │   │   ├── simulate.py
            │   │   ├── simulate_vllm.py
            │   │   ├── state_eval_v1.py
            │   │   └── utils.py
            │   ├── model/           # 模型相关
            │   │   ├── __init__.py
            │   │   └── xgb_state_model.py
            │   ├── optimizer/       # 优化器相关
            │   │   ├── __init__.py
            │   │   ├── analyze_profiler.py
            │   │   ├── communication.py
            │   │   ├── custom_process.py
            │   │   ├── experience_fine_tunning.py
            │   │   ├── global_best_custom.py
            │   │   ├── interfaces/   # 接口定义
            │   │   │   ├── __init__.py
            │   │   │   ├── benchmark.py
            │   │   │   ├── custom_process.py
            │   │   │   └── simulator.py
            │   │   ├── optimizer.py
            │   │   ├── performance_tunner.py
            │   │   ├── plugins/      # 插件系统
            │   │   │   ├── __init__.py
            │   │   │   ├── benchmark.py
            │   │   │   ├── plugin.md
            │   │   │   └── simulate.py
            │   │   ├── register.py
            │   │   ├── scheduler.py
            │   │   ├── server.py
            │   │   ├── simulator.py
            │   │   ├── store.py
            │   │   └── utils.py
            │   ├── patch/           # 补丁相关
            │   │   ├── __init__.py
            │   │   ├── mindie_plugin/
            │   │   │   ├── plugin_init_patch.py
            │   │   │   └── simulate/
            │   │   │       ├── __init__.py
            │   │   │       └── simulate_plugin.py
            │   │   ├── model_runner_patch.patch
            │   │   ├── patch_manager.py
            │   │   ├── patch_vllm.py
            │   │   └── plugin_manager_patch.patch
            │   ├── plugins/         # 插件目录
            │   │   └── __init__.py
            │   ├── sitecustomize.py
            │   └── train/           # 训练相关
            │       ├── __init__.py
            │       ├── pretrain.py
            │       ├── source_to_train.py
            │       └── state_param.py
            └── pyproject.toml       # 项目配置文件