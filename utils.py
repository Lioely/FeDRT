import os
import torch
import importlib
from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # import_module->import py file . getattr get 'cls' class from the py file.
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    # get_obj_from_str(config["target"])返回一个类A ，(**config.get("params", dict()))就是类A初始化的参数
    # get_obj_from_str(config["target"]) -> A, get_obj_from_str(config["target"])() -> A()
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
