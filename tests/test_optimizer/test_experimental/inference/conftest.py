# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tempfile
import json
from pathlib import Path
import pytest
from src.optimize.experimental.inference.file_reader import StaticFile


@pytest.fixture
def static_file():
    # 构造StaticFile
    with tempfile.TemporaryDirectory() as base_dir:
        sf = StaticFile(base_path=Path(base_dir))
        with open(sf.hardware_path, "w") as hf:
            data = {
                "cpu_count": 256,
                "cpu_mem": 2063351,
                "soc_name": "xxxx",
                "npu_mem": 62259
            }
            json.dump(data, hf)
        with open(sf.env_path, "w") as ef:
            data = {
                "atb_llm_razor_attention_enable": 0,
                "atb_llm_razor_attention_rope": 0,
                "bind_cpu": 1,
                "atb_llm_lcoc_enable": 0,
                "lccl_deterministic": 0,
                "hccl_deterministic": 0,
                "atb_matmul_shuffle_k_enable": 1,
                "mies_use_mb_swapper": 0,
                "mies_pecompute_threshold": 0.5,
                "mies_tokenizer_sliding_window_size": 0
            }
            json.dump(data, ef)
        with open(sf.mindie_config_path, "w") as mf:
            data = {
                "cache_block_size": 128,
                "max_seq_len": 10000,
                "world_size": 8,
                "cpu_mem_size": 5,
                "npu_mem_size": -1,
                "max_prefill_tokens": 6144,
                "max_prefill_batch_size": 15,
                "max_batch_size": 20
            }
            json.dump(data, mf)
        with open(sf.config_path, "w") as f:
            data = {
                "architectures": [
                    "DeepseekV3ForCausalLM"
                ],
                "hidden_act": "silu",
                "initializer_range": 0.02,
                "intermediate_size": 18432,
                "max_position_embeddings": 163840,
                "model_type": "deepseekv2",
                "num_attention_heads": 128,
                "num_hidden_layers": 6,
                "tie_word_embeddings": False,
                "torch_dtype": "float16",
                "use_cache": True,
                "vocab_size": 129280,
                "quantize": "w8a8_dynamic",
                "quantization_config": {
                    "group_size": 0,
                    "kv_quant_type": None,
                    "reduce_quant_type": None
                }
            }
            json.dump(data, f)
        with open(sf.model_struct_path, "w") as f:
            f.write("""total_param_num,total_param_size,embed_tokens_param_size_rate,self_attn_param_size_rate,"""\
                    """input_layernorm_param_size_rate,post_attention_layernorm_param_size_rate,mlp_param_size_rate,"""\
                        """norm_param_size_rate,lm_head_param_size_rate
52,1923244800,0.4818347659122749,0.003282452655013028,4.4724415737403786e-05,"""\
    """2.2362207868701893e-05,0.002862761932334355,3.727034644783649e-06,0.030114439929851883""")
        with open(sf.model_decode_op_path, "w") as f:
            f.write(
                "batch_size,max_seq_len,op_name,call_count,input_count,input_dtype,input_shape,output_count," \
                    "output_dtype,output_shape,host_setup_time,host_execute_time,kernel_execute_time," \
                        "aic_cube_fops,aiv_vector_fops")
            f.write("\n")
            f.write("""1,5,RmsNormOperation,6,5,float16;float16;float16;float16;int8,"1,7168;7168;7168;1;1",1,"""\
                    """int8,"1,7168",12.446767676767674,14.622525252525259,4.713,0.0,108480.0
1,5,ActivationOperation,3,1,float16,"1,256",1,float16,"1,128",10.030909090909093,12.30717171717172,3.07,0.0,8704.0
1,5,ActivationOperation,3,1,float16,"8,256",1,float16,"8,128",10.148282828282827,12.25,5.014,0.0,34816.0
1,5,ActivationOperation,3,1,float16,"1,2304",1,float16,"1,1152",10.787474747474748,12.65040404040404,"""\
    """5.44,0.0,39168.0""")
        with open(sf.model_prefill_op_path, "w") as f:
            f.write(
                "batch_size,max_seq_len,op_name,call_count,input_count,input_dtype,input_shape,output_count," \
                    "output_dtype,output_shape,host_setup_time,host_execute_time,kernel_execute_time," \
                        "aic_cube_fops,aiv_vector_fops")
            f.write("\n")
            f.write("""1,4,SplitOperation,6,1,float16,"6144,576",2,float16;float16,"6144,512;6144,64","""\
                    """8.658484848484846,11.53181818181819,7.253,0.0,985408.0
1,4,ElewiseOperation,15,2,float16;float16,"6144,7168;6144,7168",1,float16,"6144,7168",10.84767676767677,"""\
    """13.54727272727273,207.83,0.0,66295808.0
1,4,ActivationOperation,3,1,float16,"6144,2304",1,float16,"6144,1152",10.441616161616162,"""\
    """13.294949494949494,28.269,0.0,67324800.0""")

        yield sf
        sf.hardware_path.unlink()
        sf.env_path.unlink()
        sf.mindie_config_path.unlink()
        sf.config_path.unlink()
        sf.model_struct_path.unlink()
        sf.model_decode_op_path.unlink()
        sf.model_prefill_op_path.unlink()


@pytest.fixture
def static_file():
    # 构造StaticFile
    with tempfile.TemporaryDirectory() as base_dir:
        sf = StaticFile(base_path=Path(base_dir))
        with open(sf.hardware_path, "w") as hf:
            data = {
                "cpu_count": 256,
                "cpu_mem": 2063351,
                "soc_name": "xxxx",
                "npu_mem": 62259
            }
            json.dump(data, hf)
        with open(sf.env_path, "w") as ef:
            data = {
                "atb_llm_razor_attention_enable": 0,
                "atb_llm_razor_attention_rope": 0,
                "bind_cpu": 1,
                "atb_llm_lcoc_enable": 0,
                "lccl_deterministic": 0,
                "hccl_deterministic": 0,
                "atb_matmul_shuffle_k_enable": 1,
                "mies_use_mb_swapper": 0,
                "mies_pecompute_threshold": 0.5,
                "mies_tokenizer_sliding_window_size": 0
            }
            json.dump(data, ef)
        with open(sf.mindie_config_path, "w") as mf:
            data = {
                "cache_block_size": 128,
                "max_seq_len": 10000,
                "world_size": 8,
                "cpu_mem_size": 5,
                "npu_mem_size": -1,
                "max_prefill_tokens": 6144,
                "max_prefill_batch_size": 15,
                "max_batch_size": 20
            }
            json.dump(data, mf)
        with open(sf.config_path, "w") as f:
            data = {
                "architectures": [
                    "DeepseekV3ForCausalLM"
                ],
                "hidden_act": "silu",
                "initializer_range": 0.02,
                "intermediate_size": 18432,
                "max_position_embeddings": 163840,
                "model_type": "deepseekv2",
                "num_attention_heads": 128,
                "num_hidden_layers": 6,
                "tie_word_embeddings": False,
                "torch_dtype": "float16",
                "use_cache": True,
                "vocab_size": 129280,
                "quantize": "w8a8_dynamic",
                "quantization_config": {
                    "group_size": 0,
                    "kv_quant_type": None,
                    "reduce_quant_type": None
                }
            }
            json.dump(data, f)
        with open(sf.model_struct_path, "w") as f:
            f.write("""total_param_num,total_param_size,embed_tokens_param_size_rate,self_attn_param_size_rate,"""\
                    """input_layernorm_param_size_rate,post_attention_layernorm_param_size_rate,"""\
                        """mlp_param_size_rate,norm_param_size_rate,lm_head_param_size_rate
52,1923244800,0.4818347659122749,0.003282452655013028,4.4724415737403786e-05,2.2362207868701893e-05,"""\
    """0.002862761932334355,3.727034644783649e-06,0.030114439929851883""")
        with open(sf.model_decode_op_path, "w") as f:
            f.write(
                "batch_size,max_seq_len,op_name,call_count,input_count,input_dtype,input_shape,output_count,"\
                    "output_dtype,output_shape,host_setup_time,host_execute_time,kernel_execute_time,aic_cube_fops,"\
                        "aiv_vector_fops")
            f.write("\n")
            f.write("""1,5,RmsNormOperation,6,5,float16;float16;float16;float16;int8,"1,7168;7168;7168;1;1",1,"""\
                    """int8,"1,7168",12.446767676767674,14.622525252525259,4.713,0.0,108480.0
1,5,ActivationOperation,3,1,float16,"1,256",1,float16,"1,128",10.030909090909093,12.30717171717172,3.07,0.0,8704.0
1,5,ActivationOperation,3,1,float16,"8,256",1,float16,"8,128",10.148282828282827,12.25,5.014,0.0,34816.0
1,5,ActivationOperation,3,1,float16,"1,2304",1,float16,"1,1152",10.787474747474748,12.65040404040404,5.44,0.0,"""\
    """39168.0""")
        with open(sf.model_prefill_op_path, "w") as f:
            f.write(
                "batch_size,max_seq_len,op_name,call_count,input_count,input_dtype,input_shape,output_count,"\
                    "output_dtype,output_shape,host_setup_time,host_execute_time,kernel_execute_time,aic_cube_fops,"\
                        "aiv_vector_fops")
            f.write("\n")
            f.write("""1,4,SplitOperation,6,1,float16,"6144,576",2,float16;float16,"6144,512;6144,64","""\
                    """8.658484848484846,11.53181818181819,7.253,0.0,985408.0
1,4,ElewiseOperation,15,2,float16;float16,"6144,7168;6144,7168",1,float16,"6144,7168",10.84767676767677,"""\
    """13.54727272727273,207.83,0.0,66295808.0
1,4,ActivationOperation,3,1,float16,"6144,2304",1,float16,"6144,1152",10.441616161616162,13.294949494949494,"""\
    """28.269,0.0,67324800.0""")

        yield sf
        sf.hardware_path.unlink()
        sf.env_path.unlink()
        sf.mindie_config_path.unlink()
        sf.config_path.unlink()
        sf.model_struct_path.unlink()
        sf.model_decode_op_path.unlink()
        sf.model_prefill_op_path.unlink()
