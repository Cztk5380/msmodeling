# RFC: Hunyuanvideo模型建模支持


## 元数据
| 项目 | 内容                                        |
| :--- |:------------------------------------------|
| **状态** | 已批准                                       |
| **作者** | genius52                                  |
| **创建日期** | 2025-12-25                                |
| **相关链接** | https://gitcode.com/Ascend/msit/pull/4911 |

---

## 1. 概述
本提案旨在解决hunyuanvideo多模态模型适配问题，使其能够在tensor_cast框架下正确运行。
## 2. 方案设计
### 模型输入构造适配
为支持tensor_cast框架下的模型结构解析与图构建，需为 HunyuanVideo系列模型提供符合其前向接口的占位输入，这些输入使用 device="meta" 的张量。
当前已实现对以下两类模型的输入生成逻辑：
- HunyuanVideoTransformer3DModel：构造基础的 encoder_attention_mask；
- HunyuanVideo15Transformer3DModel：除基础 attention mask 外，还支持的 encoder_hidden_states_2 和 image_embeds，其维度由模型配置决定。

### 兼容性补丁
HunyuanVideo1.5所依赖的部分控制流逻辑（如 if tensor、tensor.item() 或 tensor[idx]）在 meta 张量上会触发运行时错误或返回无效值，导致模型构造失败。
(tensor_cast框架中，模型初始化阶段会使用 torch.device("meta") 构造“元张量”（meta tensors），以高效推导计算图和内存布局，而不实际分配显存)
为解决此问题，引入patch_torch_op上下文管理器，在模型加载期间临时修补torch.Tensor 的以下方法：
- __bool__：避免 if tensor 在 meta 张量上抛出异常，统一返回 True（仅用于控制流占位）；
- item()：防止标量提取操作失败，返回占位值 True；
- __getitem__：对 meta 张量的索引操作返回形状正确的空 meta 张量，而非执行实际索引。


## 3. 实施计划
### 已完成
- [x] 已完成hunyuanvideo的dit建模
- [x] 已完成hunyuanvideo1.5的dit建模

### 后续开发
- [ ] image输入的支持
- [ ] vae建模
- [ ] diffusers pipeline加载支持
- [ ] 各种并行特性支持