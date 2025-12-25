# RFC: Modeling Support for Hunyuanvideo Model


## Metadata
| Item | Details |
|:-----|:--------|
| **Status** | Approved |
| **Author** | wqh17101 |
| **Creation Date** | 2025-12-19 |
| **Related Links** |  https://gitcode.com/Ascend/msit/pull/4911 |

---

## 1. Oversview
This proposal aims to address the adaptation issues of the Hunyuanvideo multimodal model, enabling it to run correctly under the tensor_cast framework.
## 2. Detailed Design
### Model Input Construction Adaptation
To support model structure parsing and graph construction under the tensor_cast framework, placeholder inputs conforming to the forward interface of HunyuanVideo series models need to be provided, using tensors with device="meta".Input generation logic has been implemented for the following two types of models:
- HunyuanVideoTransformer3DModel: Constructs the basic encoder_attention_mask;
- HunyuanVideo15Transformer3DModel: In addition to the basic attention mask, it also supports encoder_hidden_states_2 and image_embeds, whose dimensions are determined by the model configuration.

### Compatibility Patch
Some control flow logic relied on by HunyuanVideo 1.5 (such as if tensor, tensor.item(), or tensor[idx]) triggers runtime errors or returns invalid values on meta tensors, leading to model construction failure.(In the tensor_cast framework, during the model initialization phase, "meta tensors" are constructed using torch.device("meta") to efficiently deduce computation graphs and memory layouts without actually allocating video memory.)To resolve this issue, the patch_torch_op context manager is introduced to temporarily patch the following methods of torch.Tensor during model loading:
- bool: Prevents if tensor from throwing exceptions on meta tensors and uniformly returns True (only used for control flow placeholder);
- item(): Prevents scalar extraction operations from failing and returns the placeholder value True;
- getitem: Returns an empty meta tensor with the correct shape for indexing operations on meta tensors, instead of performing actual indexing.


## 3. Implementation Plan
### Completed
- [x] Completed DIT modeling for hunyuanvideo
- [x] Completed DIT modeling for hunyuanvideo1.5

### Follow-up Development
- [ ] Support for image input
- [ ] VAE modeling
- [ ] Diffusers pipeline loading support
- [ ]Support for various parallelization features
