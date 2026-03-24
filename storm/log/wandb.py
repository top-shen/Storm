import os
import wandb
from dotenv import load_dotenv
from accelerate import Accelerator

load_dotenv(verbose=True)

from storm.utils import is_main_process
from storm.utils import Singleton

__all__ = [
    'WandbLogger',
    'wandb_logger'
]

class WandbLogger(metaclass=Singleton):
    def __init__(self):
        self.is_main_process = True
        self.initialized = False

    def init_logger(self, project, name, config, dir, accelerator: Accelerator = None):
        if accelerator is None:
            self.is_main_process = is_main_process()
        else:
            self.is_main_process = accelerator.is_local_main_process

        if self.is_main_process:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            wandb.init(project=project, name=name, config=config, dir=dir)
            self.initialized = True

    def log(self, log_dict):
        if self.is_main_process and self.initialized:
            wandb.log(log_dict)

    def finish(self):
        if self.is_main_process and self.initialized:
            wandb.finish()
            self.initialized = False

wandb_logger = WandbLogger()
