from transformers import AutoConfig, AutoModel


def register(model_type, model_config, model_class):
    """
    register non official transformers model into AutoModel

    :param model_type: model_type in config.json, such as deepseek_v32
    :param model_config: register model_config class
    :param model_class: register model_class
    """
    AutoConfig.register(model_type, model_config)
    AutoModel.register(model_config, model_class)
