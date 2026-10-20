"""General Face Anti-Spoofing training task for CelebA-Spoof.

This task reuses the *general* FAS components shared by the DCN and ANRL
projects while leaving their *specific* contributions in their own projects:

    - DCN specifics (not reused): jigsaw patch destruction, cutmix across
      subdomains/labels, and the patch-similarity head.
    - ANRL specifics (not reused): meta-train/meta-test optimization and the
      dual calibration constraints.

    - General components (reused here): the balanced pos/neg training loop,
      the multi-scale encoder + classifier + auxiliary depth head, adaptive
      normalization (ANRL's AdapNorm, generalized), and the shared ``common``
      infrastructure (BaseTask, test_module, transforms, dataloaders, metrics).
"""
import os
import sys
import importlib

import torch
import wandb

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from common.task import BaseTask
from common.task.fas import test_module
from common.utils import *
from common import losses


class FASBaseTask(BaseTask):
    """General image-level FAS training/evaluation task."""

    def __init__(self, args, task_dir):
        super(FASBaseTask, self).__init__(args, task_dir)
        # Keep the task device-agnostic when CUDA is unavailable (e.g. debug).
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')

    # ------------------------------------------------------------------ #
    # data / losses / forward
    # ------------------------------------------------------------------ #
    def _build_inputs(self, **kwargs):
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building dataloader')
        datasets = importlib.import_module('datasets', package=self.task_dir)

        self.train_pos = datasets.create_dataloader(self.cfg, split='train', category='pos')
        self.train_neg = datasets.create_dataloader(self.cfg, split='train', category='neg')
        self.val_pos = datasets.create_dataloader(self.cfg, split='val', category='pos')
        self.val_neg = datasets.create_dataloader(self.cfg, split='val', category='neg')
        self.test_pos = datasets.create_dataloader(self.cfg, split='test', category='pos')
        self.test_neg = datasets.create_dataloader(self.cfg, split='test', category='neg')

        self.train_loaders = [self.train_pos, self.train_neg]
        world_size = getattr(self, 'world_size', self.cfg.world_size)
        length = min(len(self.train_pos.dataset), len(self.train_neg.dataset))
        self.epoch_size = max(1, length // (self.cfg.train.batch_size * world_size))

    def _build_losses(self, **kwargs):
        self.criterion = losses.__dict__[self.cfg.loss.name](**self.cfg.loss.params).to(self.device)

        self.aux_criterion = None
        aux_cfg = getattr(self.cfg, 'aux_loss', None)
        if aux_cfg is not None and getattr(aux_cfg, 'enabled', False):
            self.aux_criterion = losses.__dict__[aux_cfg.name](**aux_cfg.params).to(self.device)

    def _model_forward(self, images):
        """Forward hook shared by train and eval.

        During training we return all outputs for loss computation; during
        eval we return only the spoof probability, as expected by
        ``common.task.fas.test_module``.
        """
        logits, aux, feat = self.model(images)
        if self.model.training:
            return logits, aux, feat
        probs = torch.softmax(logits, dim=1)[:, 1]
        return probs

    # ------------------------------------------------------------------ #
    # train / validate / test
    # ------------------------------------------------------------------ #
    def fit(self):
        if self.cfg.local_rank == 0:
            self.logger.info('=> Starting training')

        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            if self.cfg.distributed and not self.cfg.debug:
                for loader in self.train_loaders:
                    loader.sampler.set_epoch(epoch)

            self.train(epoch)
            self.validate(epoch)
            self.scheduler.step(epoch)

        if self.cfg.local_rank == 0:
            self.cfg.distributed = False
            self.test()
            wandb.finish()
            self.logger.info(self.cfg)

    def train(self, epoch):
        loss_meter = AverageMeter('train_loss', ':.5f')
        cls_meter = AverageMeter('train_cls', ':.5f')
        aux_meter = AverageMeter('train_aux', ':.5f')
        acc_meter = AverageMeter('train_acc', ':.5f')
        progress = ProgressMeter(self.epoch_size,
                                 [loss_meter, cls_meter, aux_meter, acc_meter],
                                 prefix=f'Epoch:{epoch} ')

        self.model.train()
        use_aux = self.aux_criterion is not None

        for i, (pos_batch, neg_batch) in enumerate(zip(self.train_pos, self.train_neg)):
            pos_imgs, pos_labels, pos_aux = pos_batch[0], pos_batch[1].long(), pos_batch[2]
            neg_imgs, neg_labels, neg_aux = neg_batch[0], neg_batch[1].long(), neg_batch[2]

            images = torch.cat([pos_imgs, neg_imgs], dim=0).to(self.device)  # [2B, C, H, W]
            labels = torch.cat([pos_labels, neg_labels], dim=0).to(self.device)

            logits, aux_pred, _ = self._model_forward(images)
            cls_loss = self.criterion(logits, labels)
            loss = cls_loss
            aux_loss = None

            if use_aux and aux_pred is not None:
                aux_gt = torch.cat([pos_aux, neg_aux], dim=0).to(self.device)
                aux_loss = self.aux_criterion(aux_pred.squeeze(1), aux_gt)
                loss = loss + self.cfg.aux_loss.weight * aux_loss

            acc = (logits.argmax(dim=1) == labels).float().mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.cfg.distributed:
                cls_val = reduce_tensor(cls_loss.data).item()
                acc_val = reduce_tensor(acc.data).item()
                aux_val = reduce_tensor(aux_loss.data).item() if aux_loss is not None else 0.0
            else:
                cls_val = cls_loss.item()
                acc_val = acc.item()
                aux_val = aux_loss.item() if aux_loss is not None else 0.0

            loss_val = cls_val
            if aux_loss is not None:
                loss_val = cls_val + self.cfg.aux_loss.weight * aux_val

            loss_meter.update(loss_val, images.size(0))
            cls_meter.update(cls_val, images.size(0))
            aux_meter.update(aux_val, images.size(0))
            acc_meter.update(acc_val, images.size(0))

            if self.cfg.local_rank == 0 and i % self.cfg.train.print_interval == 0:
                self.logger.info(progress.display(i))

        if self.cfg.local_rank == 0:
            wandb.log({
                'train_loss': loss_meter.avg,
                'train_cls': cls_meter.avg,
                'train_aux': aux_meter.avg,
                'train_acc': acc_meter.avg,
            }, step=epoch)

    def validate(self, epoch):
        # CelebA-Spoof evaluation is image-level, so we ask test_module for
        # per-image predictions instead of video-level averaging.
        y_preds, y_trues = test_module(self.model, [self.val_pos, self.val_neg],
                                       self._model_forward, device=self.device,
                                       distributed=True)
        metrics = self._evaluate(y_preds, y_trues, threshold='auto')

        if self.cfg.local_rank == 0:
            self._save_checkpoints(metrics, epoch, monitor_metric='AUC')
            self._log_data(metrics, epoch, prefix='val')

    def _eval_image_level(self, loaders):
        """Run image-level evaluation and return numpy arrays.

        ``test_module(..., distributed=True)`` appends per-sample scalar
        tensors, so we stack (rather than concatenate) them into a 1-D array.
        """
        y_preds, y_trues = test_module(self.model, loaders, self._model_forward,
                                       device=self.device, distributed=True)
        y_preds = torch.stack([p if torch.is_tensor(p) else torch.tensor(p)
                               for p in y_preds]).cpu().numpy()
        y_trues = torch.stack([t if torch.is_tensor(t) else torch.tensor(t)
                               for t in y_trues]).cpu().numpy()
        return y_preds, y_trues

    def test(self):
        ckpt_path = f'{self.cfg.exam_dir}/ckpts/model_best.pth.tar'
        try:
            checkpoint = torch.load(ckpt_path, weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt_path)

        state_dict = checkpoint.get('state_dict', checkpoint)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            else:
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        self.logger.info(f'resume model from {ckpt_path}')

        # Rebuild val/test loaders without DistributedSampler for full-set eval.
        self.cfg.distributed = False
        datasets = importlib.import_module('datasets', package=self.task_dir)
        self.val_pos = datasets.create_dataloader(self.cfg, split='val', category='pos')
        self.val_neg = datasets.create_dataloader(self.cfg, split='val', category='neg')
        self.test_pos = datasets.create_dataloader(self.cfg, split='test', category='pos')
        self.test_neg = datasets.create_dataloader(self.cfg, split='test', category='neg')

        # Select the operating threshold on the validation set.
        val_preds, val_trues = self._eval_image_level([self.val_pos, self.val_neg])
        metric_val = cal_metrics(val_trues, val_preds, threshold='auto')

        test_preds, test_trues = self._eval_image_level([self.test_pos, self.test_neg])
        metric = cal_metrics(test_trues, test_preds, threshold=metric_val.Thre)

        self._log_data(metric, prefix='test')


def main():
    args = get_parameters()
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = FASBaseTask(args, task_dir)
    task.prepare()
    task.fit()


if __name__ == '__main__':
    main()
