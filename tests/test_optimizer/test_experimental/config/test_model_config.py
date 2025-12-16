# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
import json
import os
from pathlib import Path

import pytest

import src.optimize.experimental.config.model_config as model_config
from src.optimize.experimental.config.model_config import ModelConfig, MindieModelConfig


@pytest.fixture
def model_config_file(tmpdir):
    _feature_file = Path(tmpdir).joinpath("llama3-8b", "config.json")
    _feature_file.parent.mkdir()
    with open(_feature_file, "w") as f:
        f.write("""
        {
          "_name_or_path": "meta-llama/Meta-Llama-3-8B",
          "architectures": [
            "LlamaForCausalLM"
          ],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 128000,
          "eos_token_id": 128001,
          "hidden_act": "silu",
          "hidden_size": 4096,
          "initializer_range": 0.02,
          "intermediate_size": 14336,
          "max_position_embeddings": 8192,
          "model_type": "llama",
          "num_attention_heads": 32,
          "num_hidden_layers": 32,
          "num_key_value_heads": 8,
          "pretraining_tp": 1,
          "rms_norm_eps": 1e-05,
          "rope_scaling": null,
          "rope_theta": 500000.0,
          "tie_word_embeddings": false,
          "torch_dtype": "float16",
          "transformers_version": "4.38.2",
          "use_cache": true,
          "vocab_size": 128256
        }
        """)
    yield _feature_file
    _feature_file.unlink()


@pytest.fixture
def mindie_config_file(model_config_file):
    _weight_path = model_config_file.parent
    _file = _weight_path.parent.joinpath("mindie", "config.json")
    _file.parent.mkdir()
    data = {
        "Version": "1.0.0",
        "ServerConfig": {
            "ipAddress": "127.0.0.1",
            "managementIpAddress": "127.0.0.2",
            "port": 8825,
            "managementPort": 8826,
            "metricsPort": 8827,
            "allowAllZeroIpListening": False,
            "maxLinkNum": 1000,
            "httpsEnabled": False,
            "fullTextEnabled": False,
            "tlsCaPath": "security/ca/",
            "tlsCaFile": [
                "ca.pem"
            ],
            "tlsCert": "security/certs/server.pem",
            "tlsPk": "security/keys/server.key.pem",
            "tlsPkPwd": "security/pass/key_pwd.txt",
            "tlsCrlPath": "security/certs/",
            "tlsCrlFiles": [
                "server_crl.pem"
            ],
            "managementTlsCaFile": [
                "management_ca.pem"
            ],
            "managementTlsCert": "security/certs/management/server.pem",
            "managementTlsPk": "security/keys/management/server.key.pem",
            "managementTlsPkPwd": "security/pass/management/key_pwd.txt",
            "managementTlsCrlPath": "security/management/certs/",
            "managementTlsCrlFiles": [
                "server_crl.pem"
            ],
            "kmcKsfMaster": "tools/pmt/master/ksfa",
            "kmcKsfStandby": "tools/pmt/standby/ksfb",
            "inferMode": "standard",
            "interCommTLSEnabled": False,
            "interCommPort": 1121,
            "interCommTlsCaPath": "security/grpc/ca/",
            "interCommTlsCaFiles": [
                "ca.pem"
            ],
            "interCommTlsCert": "security/grpc/certs/server.pem",
            "interCommPk": "security/grpc/keys/server.key.pem",
            "interCommPkPwd": "security/grpc/pass/key_pwd.txt",
            "interCommTlsCrlPath": "security/grpc/certs/",
            "interCommTlsCrlFiles": [
                "server_crl.pem"
            ],
            "openAiSupport": "vllm",
            "tokenTimeout": 600,
            "e2eTimeout": 600,
            "distDPServerEnabled": False
        },
        "BackendConfig": {
            "backendName": "mindieservice_llm_engine",
            "modelInstanceNumber": 1,
            "npuDeviceIds": [
                [
                    3
                ]
            ],
            "tokenizerProcessNumber": 8,
            "multiNodesInferEnabled": False,
            "multiNodesInferPort": 1120,
            "interNodeTLSEnabled": False,
            "interNodeTlsCaPath": "security/grpc/ca/",
            "interNodeTlsCaFiles": [
                "ca.pem"
            ],
            "interNodeTlsCert": "security/grpc/certs/server.pem",
            "interNodeTlsPk": "security/grpc/keys/server.key.pem",
            "interNodeTlsPkPwd": "security/grpc/pass/mindie_server_key_pwd.txt",
            "interNodeTlsCrlPath": "security/grpc/certs/",
            "interNodeTlsCrlFiles": [
                "server_crl.pem"
            ],
            "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
            "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
            "ModelDeployConfig": {
                "maxSeqLen": 2560,
                "maxInputTokenLen": 2048,
                "truncation": False,
                "ModelConfig": [
                    {
                        "modelInstanceType": "Standard",
                        "modelName": "llama3-8b",
                        "modelWeightPath": str(_weight_path),
                        "worldSize": 1,
                        "cpuMemSize": 5,
                        "npuMemSize": -1,
                        "backendType": "atb",
                        "trustRemoteCode": True
                    }
                ]
            },
            "ScheduleConfig": {
                "templateType": "Standard",
                "templateName": "Standard_LLM",
                "cacheBlockSize": 128,
                "maxPrefillBatchSize": 340,
                "maxPrefillTokens": 8192,
                "prefillTimeMsPerReq": 714,
                "prefillPolicyType": 0,
                "decodeTimeMsPerReq": 362,
                "decodePolicyType": 3,
                "maxBatchSize": 650,
                "maxIterTimes": 512,
                "maxPreemptCount": 121,
                "supportSelectBatch": False,
                "maxQueueDelayMicroseconds": 200499
            }
        }
    }
    with open(_file, "w") as f:
        json.dump(data, f)
    yield _file
    _file.unlink()


def mock_npu_total_memory(device_id: int = 0):
    return 65535, 3


class TestModelConfig:
    # 测试用例1: 测试文件不存在的情况
    @classmethod
    def test_init_file_not_found(cls):
        config_path = Path("non_existent_file.json")
        with pytest.raises(FileNotFoundError):
            ModelConfig(config_path)

    # 测试用例2: 测试JSON格式错误的情况
    @classmethod
    def test_init_invalid_json(cls):
        config_path = Path("invalid_json.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("{invalid_json}")
        with pytest.raises(ValueError):
            ModelConfig(config_path)
        os.remove(config_path)

    # 测试用例4: 测试正常加载配置文件的情况
    @classmethod
    def test_init_success(cls, model_config_file):
        config_instance = ModelConfig(model_config_file)
        assert config_instance.hidden_size == 4096
        assert config_instance.intermediate_size == 14336
        assert config_instance.num_attention_heads == 32
        assert config_instance.num_hidden_layers == 32
        assert config_instance.num_key_value_heads == 8
        assert config_instance.vocab_size == 128256
        assert config_instance.max_position_embeddings == 8192
        assert config_instance.kvcache_dtype_byte == 2
        assert config_instance.cache_num == 2

    # 测试用例5: 测试配置文件缺少必需参数的情况
    @classmethod
    def test_init_missing_required_param(cls):
        config_path = Path("missing_param_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(
                '{"hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12, "num_hidden_layers": 12, ' \
                '"num_key_value_heads": 12, "vocab_size": 30522}')
        with pytest.raises(ValueError):
            ModelConfig(config_path)
        os.remove(config_path)

    @classmethod
    def test_get_one_token_cache(cls, model_config_file):
        # 创建一个ModelConfig实例
        config_instance = ModelConfig(model_config_file)

        # 调用方法并验证结果
        assert config_instance.get_one_token_cache() == 131072


class TestMindieConfig:
    @classmethod
    def test_init_config_path_not_exists(cls):
        config_path = Path("non_existent_config.json")
        with pytest.raises(FileNotFoundError):
            MindieModelConfig(config_path)

    @classmethod
    def test_init_invalid_json(cls):
        config_path = Path("invalid_json_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        with pytest.raises(ValueError):
            MindieModelConfig(config_path)
        os.remove(config_path)

    @classmethod
    def test_init_success(cls, mindie_config_file, monkeypatch):
        monkeypatch.setattr(model_config, "get_npu_total_memory", mock_npu_total_memory)
        config = MindieModelConfig(mindie_config_file)
        assert config.npu_device_ids == [[3]]
        assert config.cache_block_size == 128
        assert config.max_input_length == 8192
        assert config.tp_size == 1

    @classmethod
    def test_get_npu_mem_size_with_calculated_mem_size(cls, mindie_config_file, monkeypatch):
        monkeypatch.setattr(model_config, "get_npu_total_memory", mock_npu_total_memory)
        # 模拟环境变量和配置数据
        mindie_config = MindieModelConfig(mindie_config_file)
        # 调用方法并断言结果
        result = mindie_config.get_npu_mem_size()
        expected_mem_size = (65535 * (
                100 - 3) / 100 / 1024 * 0.8 - 14356.015625 / 1024 - 4798283776 / 1024 / 1024 / 1024)
        assert result == expected_mem_size