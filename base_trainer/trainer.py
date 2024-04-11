import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from base_trainer import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_EPOCH_WISE = 2

# maximum likelihood estimation
class MLE_Engine(Engine):

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.lr_scheduler = lr_scheduler
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.scaler = GradScaler()
    
    @staticmethod
    #@profile
    def train(engine, mini_batch):
        engine.model.train()

        # gradient accumulation or only 1 
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
        engine.config.iteration_per_update == 1:
            
            # 첫 iteration : x
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()
        
        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        x, y = mini_batch.src, mini_batch.tgt[0][:,1:]

        with autocast(not engine.config.off_autocast):
            y_hat = engine.model(x, mini_batch[0][:-1])
            
            loss = engine.crit(
                # y_hat to 2d tensor
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                # y to 1d tensor
                y.contiguous().view(-1)
            )

            # loss의 업데이트 주기 조절
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)

            # gpu가 가능한 경우에만 AMP 가능
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                engine.scaler.scale(backward_target).backward()
            
            else:
                backward_target.backward()