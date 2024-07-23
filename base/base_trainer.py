import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from pathlib import Path
from numpy import inf
from logger import TensorboardWriter
import shutil

class BaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, criterion: Any, metric_ftns: List, optimizer: torch.optim.Optimizer, config: Dict):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        self._setup_monitoring()

        self.start_epoch = 1
        self.checkpoint_dir = Path(config.save_dir) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        self.best_model_state = None
        self.best_epoch = -1
        self.best_loss = float('inf')

        if config.resume:
            self._resume_checkpoint(config.resume)

    def _setup_monitoring(self):
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = float('inf') if self.mnt_mode == 'min' else float('-inf')
            self.early_stop = self.config['trainer'].get('early_stop', float('inf'))
            self.early_stop = float('inf') if self.early_stop <= 0 else self.early_stop

    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training logic for an epoch"""
        raise NotImplementedError

    def train(self):
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch, **result}
            self._log_info(log)

            best, improved = self._evaluate_performance(log) if self.mnt_mode != 'off' else (False, False)
            if improved:
                not_improved_count = 0
                self.best_model_state = {
                    'arch': type(self.model).__name__,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'monitor_best': self.mnt_best,
                    'config': self.config
                }
                self.best_epoch = epoch
                self.best_loss = log[self.mnt_metric]
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info(f"Validation performance didn't improve for {self.early_stop} epochs. Training stops.")
                break

            if self._should_save_checkpoint(epoch, not_improved_count):
                self._save_checkpoint(epoch, log[self.mnt_metric])

        self.logger.info(f"Training completed. Best {self.mnt_metric} was {self.mnt_best} at epoch {self.best_epoch}.")
        if self.best_model_state:
            self._save_best_model()

    def _log_info(self, log: Dict[str, Any]):
        for key, value in log.items():
            self.logger.info(f'    {str(key):15s}: {value}')

    def _evaluate_performance(self, log: Dict[str, float]) -> (bool, bool):
        try:
            improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
        except KeyError:
            self.logger.warning(f"Metric '{self.mnt_metric}' not found. Model performance monitoring is disabled.")
            self.mnt_mode = 'off'
            return False, False

        best = False
        if improved:
            self.mnt_best = log[self.mnt_metric]
            best = True

        return best, improved

    def _should_save_checkpoint(self, epoch: int, not_improved_count: int) -> bool:
        return (self.save_period == 0 and not_improved_count == 0) or \
               (self.save_period != 0 and (epoch % self.save_period == 0 or not_improved_count == 0))

    def _save_checkpoint(self, epoch: int, val_loss: float):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = self.checkpoint_dir / f'e{epoch:04d}-l{val_loss:.5f}.pth'
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")

    def _save_best_model(self):
        best_path = self.checkpoint_dir / f'e{self.best_epoch:04d}-l{self.best_loss:.5f}.pth'
        save_path = self.checkpoint_dir.parent / f'best_model.pth'
        shutil.copyfile(best_path, save_path)
        self.logger.info(f"Saving current best model: {save_path} ...")

    def _resume_checkpoint(self, resume_path: str):
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        self._load_state_dict(checkpoint)
        self._load_optimizer_state(checkpoint)

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

    def _load_state_dict(self, checkpoint: Dict):
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Architecture configuration given in config file is different from that of checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

    def _load_optimizer_state(self, checkpoint: Dict):
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
