from typing import Optional
import os
import argparse
import numpy as np
import copy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
import pipeline

from torchvision.utils import save_image
from torchvision.models import vgg16
output_dir = 'logs'
version_name='Baseline'


def _seed_worker(worker_id):
    """Deterministic DataLoader worker — seeds numpy & python random per worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
import matplotlib.pyplot as plt
# import tent
import math
from pretraining.dcg import DCG as AuxCls
from model import *
from utils import *


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr

        
        config_path = r'option/diff_DDIM.yaml'
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        config = EasyDict(params)
        self.diff_opt = config

        self.model = ConditionalModel(self.params, guidance=self.params.diffusion.include_guidance)
        self.aux_model = AuxCls(self.params)
        self.init_weight(ckpt_path=getattr(self.params.model, 'aux_ckpt_path', None))
        self.aux_model.eval()

        self.save_hyperparameters()
        
        self.gts = []
        self.preds = []

        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train'),
        )
        self.DiffSampler.scheduler.set_timesteps(self.diff_opt['scheduler']['num_test_timesteps'])
        self.DiffSampler.scheduler.diff_chns = self.params.data.num_classes

    def configure_optimizers(self):
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, self.model.parameters()))
        # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.99],weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]


    def init_weight(self,ckpt_path=None):
        
        if ckpt_path and os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path,map_location=self.device)[0]
            checkpoint_model = checkpoint
            state_dict = self.aux_model.state_dict()
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(checkpoint_model.keys())
            state_dict.update(checkpoint_model)
            
            self.aux_model.load_state_dict(state_dict) 
        elif ckpt_path:
            print(f"Auxiliary checkpoint not found, training with random aux weights: {ckpt_path}")

    def diffusion_focal_loss(self, prior, targets, noise, noise_gt, gamma=1, alpha=10):
        probs = F.softmax(prior, dim=1)
        probs = (probs * targets).sum(dim=1)
        weights = 1+alpha*(1 - probs) ** gamma
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        loss = weights*(noise-noise_gt).square()
        return loss.mean()



    def guided_prob_map(self, y0_g, y0_l, bz, nc, np):
    
        distance_to_diag = torch.tensor([[abs(i-j)  for j in range(np)] for i in range(np)]).to(self.device)

        weight_g = 1 - distance_to_diag / (np-1)
        weight_l = distance_to_diag / (np-1)
        interpolated_value = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        diag_indices = torch.arange(np)
        map = interpolated_value.clone()
        for i in range(bz):
            for j in range(nc):
                map[i,j,diag_indices,diag_indices] = y0_g[i,j]
                map[i,j, np-1, 0] = y0_l[i,j]
                map[i,j, 0, np-1] = y0_l[i,j]
        return map

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.aux_model.eval()
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        #bicubic = bicubic.cuda()
        x_batch = x_batch.cuda()
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
            # y0_aux_global,y0_aux_local = y0_aux_global.softmax(1),y0_aux_local.softmax(1)
        # loss_aux = self.aux_cost_function(y0_aux,y_batch)
        # loss_aux.backward()
        
        
        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()
        
        y_map = y_batch.unsqueeze(1).expand(-1,np*np,-1).reshape(bz*np*np,nc)
        noise = torch.randn_like(y_map).to(self.device)
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bz*np*np,), device=self.device).long()

        noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
        noisy_y = noisy_y.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        
        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        y_fusion = torch.cat([y0_cond, noisy_y],dim=1)

        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns)

        noise = noise.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        loss = self.diffusion_focal_loss(y0_aux,y_batch,noise_pred,noise)

        self.log("train_loss",loss,prog_bar=True)
        return {"loss":loss}

    # def validation_step_end(self,step_output):
    #     model_state_dict = self.model.state_dict()
    #     torch.save(model_state_dict, os.path.join(self.save_path,'ckp.pth'))
    #     print('checkpoint save!')
    #     ema_model_state_dict = self.ema_model.state_dict()
    #     for key in model_state_dict:
    #         ema_model_state_dict[key] = 0.999*ema_model_state_dict[key] + 0.001*model_state_dict[key]
    #     self.ema_model.load_state_dict(ema_model_state_dict)
    def on_validation_epoch_end(self):
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

        self.log('accuracy',ACC)
        self.log('f1',F1)
        self.log('Precision',Prec)        
        self.log('Recall',Rec)
        self.log('AUC',AUC_ovo)
        self.log('kappa',kappa)
        try:
            import numpy as _np
            _gt_np = gt.detach().cpu().numpy()
            _pr_np = pred.detach().cpu().numpy()
            _y_true = _np.argmax(_gt_np, axis=1).astype(int)
            _exp = _np.exp(_pr_np - _pr_np.max(axis=1, keepdims=True))
            _y_score = _exp / _exp.sum(axis=1, keepdims=True)
            _pred_path = os.environ.get('DIFFMICV2_PRED_PATH',
                                        os.path.join(output_dir, 'predictions_val.npz'))
            os.makedirs(os.path.dirname(_pred_path) or '.', exist_ok=True)
            _np.savez(_pred_path, y_true=_y_true, y_score=_y_score,
                       acc=float(ACC), bacc=float(BACC), f1=float(F1), auc=float(AUC_ovo))
            print(f"[save-predictions] wrote: {_pred_path}  (N={_y_true.shape[0]})")
        except Exception as _e:
            print(f"[save-predictions] failed: {_e}")
        self.gts = []
        self.preds = []
        print("Val: Accuracy {0}, F1 score {1}, Precision {2}, Recall {3}, AUROC {4}, Cohen Kappa {5}".format(ACC,F1,Prec,Rec,AUC_ovo,kappa))


    def validation_step(self,batch,batch_idx):
        self.model.eval()
        self.aux_model.eval()

        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()

        
        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        # yT = torch.rand_like(y0_cond)
        yT = self.guided_prob_map(torch.rand_like(y0_aux_global),torch.rand_like(y0_aux_local),bz,nc,np)
        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        y_pred = self.DiffSampler.sample_high_res(x_batch,yT,conditions=[y0_cond, patches, attns])
        y_pred = y_pred.reshape(bz, nc, np*np)
        y_pred = y_pred.mean(2)
        self.preds.append(y_pred)
        self.gts.append(y_batch)

        
        # self.log('accuracy',ACC)
        # self.log('f1',F1)
        
        # return {"gt":y_batch,"pred":y_pred}
    
    def train_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)
        _use_balanced = os.environ.get('DIFFMICV2_BALANCED_SAMPLER', '0') == '1'
        _use_weighted = os.environ.get('DIFFMICV2_WEIGHTED_SAMPLER', '0') == '1'
        if _use_balanced:
            import sys as _sys, os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            _tools = _os.path.normpath(_os.path.join(_here, '..', 'tools'))
            if _tools not in _sys.path:
                _sys.path.insert(0, _tools)
            from balanced_sampler import BalancedBatchSampler
            _labels = [int(s['label']) for s in train_dataset.data_list]
            _bsampler = BalancedBatchSampler(
                _labels, batch_size=self.params.training.batch_size,
                num_classes=self.params.data.num_classes,
                seed=int(os.environ.get("SEED", getattr(self.params.data, 'seed', 42))))
            print(f"[DiffMICv2 balanced-sampler] activated, per_class={_bsampler.per_class}")
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=_bsampler,
                num_workers=self.params.data.num_workers,
                worker_init_fn=_seed_worker,
            )
        elif _use_weighted:
            import numpy as _np
            from torch.utils.data import WeightedRandomSampler as _WRS
            _labels = [int(s['label']) for s in train_dataset.data_list]
            _cnt = _np.bincount(_labels, minlength=self.params.data.num_classes)
            _w = 1.0 / _np.maximum(_cnt, 1)
            _sw = _w[_labels]
            _sampler = _WRS(_sw.tolist(), num_samples=len(_labels), replacement=True)
            print(f"[DiffMICv2 weighted-sampler] class_count={_cnt.tolist()}  class_weight={_w.tolist()}")
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.params.training.batch_size,
                sampler=_sampler,
                num_workers=self.params.data.num_workers,
                worker_init_fn=_seed_worker,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.params.training.batch_size,
                shuffle=True,
                num_workers=self.params.data.num_workers,
                worker_init_fn=_seed_worker,
            )
        return train_loader
    
    def val_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
            worker_init_fn=_seed_worker,
        )
        return test_loader


def parse_args():
    parser = argparse.ArgumentParser(description='DiffMICv2 trainer')
    parser.add_argument('--config', default='configs/placental.yml', help='training config yaml')
    parser.add_argument('--resume', default=None, help='checkpoint path to resume')
    parser.add_argument('--cpu', action='store_true', help='train on CPU')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='early stopping patience in validation epochs (0 disables)')
    return parser.parse_args()


def main():
    args = parse_args()
    resume_checkpoint_path = args.resume

    config_path = args.config
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    _cfg_seed = getattr(getattr(config, 'data', None), 'seed', 42)
    seed = int(os.environ.get("SEED", _cfg_seed))
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    print(f"[seed] effective seed = {seed} (env SEED > config.data.seed > 42)")
    logger = TensorBoardLogger(name=os.path.splitext(os.path.basename(config_path))[0], save_dir=output_dir)

    model = CoolSystem(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='AUC',
        filename='placental-epoch{epoch:02d}-accuracy-{accuracy:.4f}-f1-{f1:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=1,
        mode = "max",
        save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=config.training.n_epochs,
        accelerator='cpu' if args.cpu else 'gpu',
        devices=1,
        precision=32,
        logger=logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=5,
        deterministic=True,
        callbacks = [checkpoint_callback,lr_monitor_callback] + (
            [EarlyStopping(monitor='AUC', mode='max',
                           patience=max(1, int(args.early_stop_patience) // 5))]
            if args.early_stop_patience and args.early_stop_patience > 0 else []
        )
    )

    trainer.fit(model,ckpt_path=resume_checkpoint_path)

    _best_ckpt = checkpoint_callback.best_model_path
    if _best_ckpt and os.path.isfile(_best_ckpt):
        print(f"[post-train] best ckpt → {_best_ckpt}")
        trainer.validate(model, ckpt_path=_best_ckpt)
        print(f"[post-train] val predictions saved → {os.environ.get('DIFFMICV2_PRED_PATH', 'predictions_val.npz')}")
    else:
        print(f"[post-train] WARNING: best ckpt not found — val predictions may be from last epoch")
    
if __name__ == '__main__':
    main()
