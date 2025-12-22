# RFC: General Model and Configuration Loading Optimization Proposal

## Metadata

| Item | Details |
|:-----|:--------|
| **Status** | Approved |
| **Author** | wqh17101 |
| **Creation Date** | 2025-12-19 |
| **Related Links** | [1. Optimize model and config loading logic 2. Add model_type support for mapping (remove model_id mapping later)](https://gitcode.com/Ascend/msit/pull/4845)<br/><br/>[Add Xiaomi model loading, fix reload config logic & adaptive LMHead addition & DT synchronization & optimize quantization logic](https://gitcode.com/Ascend/msit/pull/4880) |

---

## 1. Overview

This proposal aims to address the insufficient capabilities in model loading and general configuration loading within the project. The solution focuses on optimizing the architecture and configuration, removing redundant configurations, adopting adaptive methods for automatic configuration whenever possible, and maximizing the reuse of transformers library capabilities.

## 2. Detailed Design

- To ensure single responsibility, we designed an independent `AutoModelConfigLoader` class to implement the functions of loading models, loading general configurations.
- For model structure registration and mapping, `model_type` should be used as the key instead of `model_id`.
- `ModelConfig` refactoring

### 2.1 Implementation Plan

```mermaid
graph TD
    A[User Input Config] --> C[ConfigResolver]
    B[Model Native Config] --> C
    C --> D[Runtime Config]
    A1[Model Path/ID] --> A
    A3[Parallel Settings] --> A
    A4[Compilation Options] --> A
    B1[Model Structure Parameters] --> B
    B2[Attention Mechanism Type] --> B
    B3[Special Module Configurations] --> B
    D2[Final Parallel Config] --> D
    D3[Final Model Structure Config] --> D
    D4[Runtime Optimization Config] --> D
```

#### 2.1.1 General Configuration Files

For standard `config.json`, we use `AutoConfig.from_pretrained` method for reading.

```mermaid
graph TD
    A[Start: load_config method] --> B[Call check_model_path to verify model path]
    B --> C{Check Result: Only config.json exists?}
    C -->|Yes| D[Update model_id to full path of config.json]
    C -->|No| E[Keep original model_id]
    D --> F[Attempt to load config using native Transformers]
    E --> F
    F --> G{Load successful?}
    G -->|Yes| H[Set is_transformers_natively_supported = True]
    G -->|No| I[Reload with trust_remote_code=True]
    H --> J[Log: is_transformers_natively_supported status]
    I --> K[Check if instantiated model_type matches config <br> e.g. kimi_k2's actual model_type is deepseek]
    K --> L{model_type differs?}
    L -->|Yes| M[Reload config using actual model_type]
    L -->|No| N[Keep current config]
    M --> O[Set is_transformers_natively_supported = True]
    N --> P[Set is_transformers_natively_supported = False]
    O --> J
    P --> J
    J --> Q[Return hf_config]
```

#### 2.1.2 General Model Loading

We use `AutoModel` or `AutoModelForCausalLM` for loading, where `AutoModelForCausalLM = AutoModelWithLMHead`.

```mermaid
graph TD
    A[Start: load_model method] --> B[Receive parameters: hf_config, dtype, **kwargs]
    B --> C[Determine trust_remote_code value]
    C --> D{trust_remote_code in kwargs?}
    D -->|Yes| E[Use trust_remote_code from kwargs]
    D -->|No| F[trust_remote_code = not is_transformers_natively_supported]
    E --> G[Call try_to_load_model method]
    F --> G
    G --> H[Attempt to load model using AutoModel.from_config]
    H --> I{Load successful?}
    I -->|Yes| J[Return AutoModel instance]
    I -->|No| K[Catch exception]
    K --> L[Attempt to load model using AutoModelForCausalLM.from_config]
    L --> M{Load successful?}
    M -->|Yes| N[Return AutoModelForCausalLM instance]
    M -->|No| O[Throw exception]
    J --> P[End: Return model instance]
    N --> P
```

### 2.2 Alternative Solutions

1. **Maintain Status Quo**: Continue managing model and config loading functions across various modules
   - **Disadvantages**: Will lead to more circular dependency issues, difficult to maintain and extend

2. **Use Inheritance Instead of Composition**: Extend model loading functionality through inheritance
   - **Disadvantages**: Increases complexity of class hierarchy, less flexible

### 2.3 Solution Analysis

#### Advantages of Proposed Solution:

1. Solves circular dependency issues between modules, improving code quality
2. Improves model type recognition, enhancing system compatibility
3. Follows single responsibility principle, improving code maintainability
4. Adopts layered architecture design, facilitating extension and maintenance
5. Supports configuration-driven approach, enhancing system flexibility

#### Limitations of Proposed Solution:

1. Requires updating existing model and config loading usage patterns
2. Adds new modules, requiring corresponding documentation and training
3. Requires large-scale refactoring of existing code

## 3. Implementation Plan

### General config and model loading refactoring

- [x] Extract a model loader class for responsibility separation
- [x] Support model loading for various scenarios
- [ ] Use model_type instead of model_id as the key for model structure mapping dictionary

### ModelConfig refactoring

- [x] Remove enable_lmhead
- [x] Remove disable_auto_map
- [ ] Remove hf_config_json
- [ ] Continue optimization based on changes

### User Interaction Refactoring

- [ ] Continue optimization based on changes

---

## Technical Implementation Details

### Core Components

#### AutoModelConfigLoader
This class serves as the central hub for all configuration and model loading operations:

- **Configuration Loading**: Handles various configuration formats and sources
- **Model Loading**: Supports different model architectures and loading strategies

### Key Design Principles

1. **Single Responsibility**: Each component has a clear, focused purpose
2. **Extensibility**: New model architectures can be easily integrated
3. **Compatibility**: Works synergistically with existing transformers library features
4. **Performance**: Optimized for production environments
5. **Maintainability**: Clear separation of concerns reduces complexity

### Migration Strategy

Implementation follows a phased approach:
1. Core infrastructure setup
2. Configuration system unification
3. Model loading integration
4. User interface optimization
5. Performance validation and tuning

This RFC represents a significant architectural improvement that will enhance system flexibility, maintainability, and performance while providing better support for different model types.