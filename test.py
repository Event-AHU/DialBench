import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import bliva.tasks as tasks
from bliva.common.config import Config
from bliva.common.dist_utils import get_rank, init_distributed_mode
from bliva.common.logger import setup_logger
from bliva.common.registry import registry
from bliva.datasets.builders import *
from bliva.models import *
from bliva.processors import *
from bliva.runners import *
from bliva.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.")
    parser.add_argument("--checkpoint", required=True, help="path to model checkpoint.")
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_runner_class(cfg):
    return registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

def main():
    args = parse_args()
    cfg = Config(args)
    
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    
    cfg.pretty_print()
    
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    runner_class = get_runner_class(cfg)
    runner = runner_class(
        cfg=cfg, task=task, model=model, datasets=datasets
    )
    
    # Perform evaluation
    eval_results = runner.evaluate(skip_reload=True)
    
    # Print results
    for k, v in eval_results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()