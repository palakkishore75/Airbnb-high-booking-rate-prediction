import yaml
import importlib

def get_models_from_config(config_path='config/model_config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    models = {}
    for name, model_info in config.items():
        module_name, class_name = model_info['model_class'].rsplit('.', 1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = model_class(**model_info['params'])
        models[name] = model

    return models