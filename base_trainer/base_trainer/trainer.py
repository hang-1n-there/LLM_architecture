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
VERBOSE_BATCH_WISE = 2

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

        # gradient accumulation or only 1, but first iteration is not amp 
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
        
        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
        engine.state.iteration > 0:
            torch_utils.clip_grad_norm(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )

            if engine.config.gpu_id >=0 and not engine.config.off_autocast:
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()
        
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss' : loss,
            'ppl' : ppl,
            '|param|' : p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|' : g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }
    
    @staticmethod
    def validate(engine, mini_batch):
        with torch.no_grad:
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:,1:]

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch.tgt[0][:,:-1])

                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1)
                )
        
        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss' : loss,
            'ppl' : ppl
        }
    
    @staticmethod
    def attach(
        train_engine, validation_enigne,
        training_metric_names = ['loss', 'ppl','|param|','|g_param|'],
        validation_metric_names = ['loss', 'ppl'],
        verbose=VERBOSE_BATCH_WISE
    ):
        
        # 메트릭의 평균을 엔진에 첨부
        # 모델의 파라미터나 손실 폭을 줄여주기도 함.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )
        
        for metirc_name in training_metric_names:
            attach_running_average(train_engine, metirc_name)
        
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    np.exp(avg_loss),
                ))
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_enigne,metric_name)
        
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_enigne,validation_metric_names)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_enigne.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))
    
    # 학습이 멈춘 epoch 계산
    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch-1) * len(engine.state.dataloader)
        engine.state.epoch = resume_epoch - 1

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
    
    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # Set a filename for last epoch
        model_fn = config.model_fn.split('.')

        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + model_fn[[-1]]
        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )

class SingleTrainer():
    
    def __init__(self, target_engine_clcass, config):
        self.target_engine_class = target_engine_clcass
        config = config
    
    def train(self, model, crit, optimizer, train_loader, 
              valid_loader, src_vocab, tgt_vocab, n_epochs,
              lr_scheduler=None
    ):
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )

        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # train 1 epoch이 끝나면 valid 시작
        def run_validation(engine,validation_engine,valid_loader):
            validation_engine.run(valid_loader,max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()
        
        train_engine.add_envent_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine,
            valid_loader
        )

        # 학습이 처음 시작했을 때
        train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch,
        )

        validation_engine.add_envent_handler(
            Events.EPOCH_COMPLETED, self.target_engine_class.check_best
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        train_engine.run(train_loader, max_epochs=n_epochs)

        return model
    
