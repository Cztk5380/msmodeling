# RFC: Support Classifier-free Guidance Parallel for Diffusers Model Proposal

## Metadata

| Item | Details |
|:-----|:--------|
| **Status** | Approved |
| **Author** | zhengxinqian |
| **Creation Date** | 2025-12-26 |
| **Related Links** | [【feature】support classifier-free guidance parallel](https://gitcode.com/Ascend/msit/pull/4914) |

---

## 1. Overview

This proposal aims to support classifier-free guidance (CFG) feature and CFG parallel in diffusion model.

## 2. Background

Classifier-free guidance (CFG) is a feature in diffusion model inference pipeline. It enables conditional control over outputs without relying on pre-trained classifiers, simplifying the inference workflow effectively.
CFG parallel dispatch conditional control and unconditional control to two DiT model instances as input, and gather the output of DiT model.
CFG is a feature of diffusion model inference pipeline but not a part of DiT model, therefore this feature will be built in video_generate.py.

## 3. Detailed Design


- case 1 (use cfg, cfg world size == 1)
    
    Do DiT model inference twice

- case 2 (use cfg, cfg world size == 2)
    
    Do DiT model inference once, and all-gather the output.


<!-- - Split input for Dit.
- Add communication in attention.

### 2.1 Proposed solution
#### Part 1: Split input for Dit.
It is judged to be segmented on the h or w dimensions. Finally, the result of Dit needs to be all_gather on the segmentation dimension to get a complete output.
To achieve this, we modify the code as follows:
1. add ulysses-size in Parallel_Config.
2. add `process_input` in `video_generation.py`
3. add `all_gather` after dit forward if use ulysses.

#### Part 2: Add communication in attention.
To support ulysses parallelism in the attention layer, we need to modify the forward function of the attention layer.
1. add `get_sp_group` to manager communication.
2. add `all_to_all` in attention layer.

As for the communication in attention layer, we assume the input size is [b, s, head_num, head_dim], p is the ulysses-size.
After `all_to_all`, we get each part is [b, s * p, head_num, head_dim / p].
So the communication cost is `input_len / p * (p  - 1)`.

## 3. Plan
**TODO LIST**
- [ ] Add extra checks in ut to make sure ulysses parallelism works. -->