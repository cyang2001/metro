import os
import sys
import subprocess
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import dotenv
from utils.utils import get_logger, ensure_dir
dir_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(dir_path, "pc_environment.env")
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path, override=True)
@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log = get_logger(__name__)
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    mode_name = cfg.mode.get("name", "")
    dispatch_dict = {
        "train": "src.pipeline.train_pipeline:main",
        "test": "src.pipeline.test_pipeline:main",
        "demo": "src.pipeline.demo_pipeline:main"
    }

    if not mode_name:
        log.error("Please specify mode.name in the configuration (train, test, or demo)")
        sys.exit(1)
        
    if mode_name not in dispatch_dict:
        log.error(f"Invalid mode: {mode_name}. Supported modes: {list(dispatch_dict.keys())}")
        sys.exit(1)
        

    output_dir = cfg.get("output_dir", "results") 
    ensure_dir(output_dir)
    log.info(f"Output directory: {output_dir}")

    import importlib
    path_func_str = dispatch_dict[mode_name] 
    module_path, func_name = path_func_str.split(":")

    try:
        log.info(f"Loading module: {module_path}")
        mod = importlib.import_module(module_path)
    except ImportError as e:
        log.error(f"Failed to import module {module_path}: {e}")
        sys.exit(1)
        
    try:
        main_func = getattr(mod, func_name)
        if not callable(main_func):
            log.error(f"{module_path}.{func_name} is not callable")
            sys.exit(1)
            
        log.info(f"Executing {path_func_str} in {mode_name} mode")
        main_func(cfg)
    except AttributeError as e:
        log.error(f"Function {func_name} not found in module {module_path}: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error executing {path_func_str}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 