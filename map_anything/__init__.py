import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

def resolve_special_float(value):
    if value == "inf":
        return np.inf
    elif value == "-inf":
        return -np.inf
    else:
        raise ValueError(f"Unknown special float value: {value}")


def model_config() -> DictConfig:
    """
    Initialize a model using OmegaConf configuration.
    """
    if not OmegaConf.has_resolver("special_float"):
        OmegaConf.register_new_resolver("special_float", resolve_special_float)

    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="./configs")  # relative to current file
    # Compose a config by name
    cfg = hydra.compose(config_name="mapanything_pixio")
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    model_dict = OmegaConf.to_container(cfg.model_config, resolve=True)
    return model_dict
