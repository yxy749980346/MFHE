import os
import os.path as osp
from typing import Tuple, Dict

import torch
import tqdm


from pareGeo.engine.base_trainer import BaseTrainer
from pareGeo.utils.torch import to_cuda, release_cuda, validation_sample_count, reduce_validation_summary
from pareGeo.utils.summary_board import SummaryBoard
from pareGeo.utils.timer import Timer
from pareGeo.utils.common import get_log_string
from pareGeo.utils.data import precompute_neibors


class EpochBasedTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_epoch,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_epoch = max_epoch

    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        pass

    def before_val_epoch(self, epoch) -> None:
        pass

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.check_invalid_gradients():
            raise FloatingPointError(f'Non-finite gradients at epoch {epoch}, iteration {iteration}.')

    def check_loss(self, loss):
        finite = torch.isfinite(loss.detach()).all().to(dtype=torch.int32)
        if self.distributed:
            # All ranks must agree before any rank enters backward collectives.
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Non-finite loss on at least one training rank.')

    def train_epoch(self):
        if self.distributed and hasattr(self.train_loader, 'sampler'):
            self.train_loader.sampler.set_epoch(self.epoch)
        self.optimizer.zero_grad()
        self.summary_board.reset_all()
        self.timer.reset()
        try:
            self.before_train_epoch(self.epoch)
            total_iterations = len(self.train_loader)
            if total_iterations == 0:
                raise ValueError('Training data loader is empty.')
            for iteration, data_dict in enumerate(self.train_loader):
                self.inner_iteration = iteration + 1
                self.iteration += 1
                data_dict = to_cuda(data_dict)
                data = precompute_neibors(
                    data_dict['points'], data_dict['lengths'],
                    self.cfg.backbone.num_stages, self.cfg.backbone.num_neighbors,
                )
                data_dict.update(data)
                self.before_train_step(self.epoch, self.inner_iteration, data_dict)
                self.timer.add_prepare_time()
                output_dict, result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)
                self.check_loss(result_dict['loss'])

                # Average each accumulation group, including a shorter final group.
                group_start = iteration // self.grad_acc_steps * self.grad_acc_steps
                group_size = min(self.grad_acc_steps, total_iterations - group_start)
                (result_dict['loss'] / group_size).backward()
                self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
                self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
                self.optimizer_step(self.inner_iteration, force=self.inner_iteration == total_iterations)

                self.timer.add_process_time()
                self.after_train_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
                result_dict = self.release_tensors(result_dict)
                self.summary_board.update_from_result_dict(result_dict)
                if self.inner_iteration % self.log_steps == 0:
                    summary_dict = self.summary_board.summary()
                    message = get_log_string(
                        result_dict=summary_dict,
                        epoch=self.epoch,
                        max_epoch=self.max_epoch,
                        iteration=self.inner_iteration,
                        max_iteration=total_iterations,
                        lr=self.get_lr(),
                        timer=self.timer,
                    )
                    self.logger.info(message)
                    self.write_event('train', summary_dict, self.iteration)
                torch.cuda.empty_cache()
            self.after_train_epoch(self.epoch)
        except Exception:
            self.optimizer.zero_grad()
            self.logger.error(f'Training aborted at epoch {self.epoch}, iteration {self.inner_iteration}.')
            raise

        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
        self.logger.critical(message)
        if self.scheduler is not None:
            self.scheduler.step()
        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
        if not self.save_all_snapshots:
            last_snapshot = osp.join(self.snapshot_dir, f'epoch-{self.epoch - 1}.pth.tar')
            if osp.exists(last_snapshot):
                os.remove(last_snapshot)

    def inference_epoch(self):
        self.set_eval_mode()
        try:
            self.before_val_epoch(self.epoch)
            summary_board = SummaryBoard(adaptive=True)
            timer = Timer()
            total_iterations = len(self.val_loader)
            if total_iterations == 0:
                raise ValueError('Validation data loader is empty.')
            real_samples = total_iterations
            if self.distributed:
                if self.val_loader.batch_size != 1 or self.val_loader.drop_last:
                    raise ValueError('Distributed validation requires batch_size=1 and drop_last=False.')
                real_samples = validation_sample_count(self.val_loader)
            metric_totals = {}
            sample_count = 0
            pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations, ncols=180)
            for iteration, data_dict in pbar:
                self.inner_iteration = iteration + 1
                data_dict = to_cuda(data_dict)
                data = precompute_neibors(
                    data_dict['points'], data_dict['lengths'],
                    self.cfg.backbone.num_stages, self.cfg.backbone.num_neighbors,
                )
                data_dict.update(data)
                self.before_val_step(self.epoch, self.inner_iteration, data_dict)
                timer.add_prepare_time()
                output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
                self.check_loss(result_dict['loss'])
                torch.cuda.synchronize()
                timer.add_process_time()
                self.after_val_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
                result_dict = release_cuda(result_dict)
                for name in result_dict:
                    metric_totals.setdefault(name, 0.0)
                # DistributedSampler adds repeated indices after the real samples.
                # Forward them to keep DDP collectives aligned, but do not count them.
                if iteration < real_samples:
                    sample_count += 1
                    for name, value in result_dict.items():
                        metric_totals[name] += value
                    summary_board.update_from_result_dict(result_dict)
                message = get_log_string(
                    result_dict=summary_board.summary(),
                    epoch=self.epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    timer=timer,
                )
                pbar.set_description(message)
                torch.cuda.empty_cache()
            self.after_val_epoch(self.epoch)
            summary_dict = summary_board.summary()
            if self.distributed:
                device = next(self.model.parameters()).device
                summary_dict = reduce_validation_summary(metric_totals, sample_count, device)
            message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
            self.logger.critical(message)
            self.write_event('val', summary_dict, self.epoch)
        finally:
            self.set_train_mode()

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:
            # Determine which checkpoint to load
            if self.args.epoch is not None:
                # Load a specific epoch checkpoint
                snapshot_path = osp.join(self.snapshot_dir, f'epoch-{self.args.epoch}.pth.tar')
            else:
                # Load the latest snapshot (contains optimizer & scheduler state)
                snapshot_path = osp.join(self.snapshot_dir, 'snapshot.pth.tar')

            if osp.isfile(snapshot_path):
                self.load_snapshot(snapshot_path, require_training_state=True)
                self.logger.info(f'Resumed from epoch {self.epoch}, iteration {self.iteration}')
            else:
                raise FileNotFoundError(
                    f'Cannot resume: snapshot not found at "{snapshot_path}". '
                    f'Available files in {self.snapshot_dir}: {os.listdir(self.snapshot_dir) if osp.isdir(self.snapshot_dir) else "dir not found"}'
                )
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)

        self.set_train_mode()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            self.inference_epoch()
