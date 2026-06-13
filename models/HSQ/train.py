import argparse
import logging
import os
import random

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



from timm.utils import  ModelEmaV3

from models.LivNet import LENet, LENet_base
from utils.loss import FocalLoss

from utils.train_eval import evaluate, evaluate_base, train_one_epoch, train_one_epoch_base, metric_info
from timm.data import Mixup
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score ,Specificity
from tqdm import tqdm

def main(rank, k_fold, args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    args.model = args.model_name
    # Shell (train.sh) applies the fold offset via SEED env var when called
    # per-fold. In legacy all-in-one mode (no SEED env), fall back to
    # args.seed + k_fold so each fold still gets a distinct seed.
    _env_seed = os.environ.get("SEED")
    if _env_seed is not None:
        seed = int(_env_seed)
    else:
        seed = int(args.seed) + int(k_fold)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(f"[seed] use_deterministic_algorithms(warn_only=True)")
    except Exception as _e:
        print(f"[seed] use_deterministic_algorithms unavailable: {_e}")
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    _dl_generator = torch.Generator()
    _dl_generator.manual_seed(seed)
    print(f"[seed] fixed at {seed} (torch+cuda+np+random, cudnn deterministic)")
    
    _train_path = getattr(args, 'train_path', None)
    _val_path   = getattr(args, 'val_path',   None)
    _test_path  = getattr(args, 'test_path',  None)

    criterion_focal = FocalLoss(alpha=0.25, gamma=2)
    if args.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        raise Exception("Please replace it with cross entropy")

    from torchvision import datasets as _tv_datasets, transforms as _T
    _MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
    _train_tf = _T.Compose([
        _T.RandomResizedCrop(224),
        _T.RandomHorizontalFlip(p=0.4),
        _T.ToTensor(),
        _T.Normalize(_MEAN, _STD),
    ])
    _test_tf = _T.Compose([
        _T.Resize((224, 224)),
        _T.ToTensor(),
        _T.Normalize(_MEAN, _STD),
    ])
    class _ImageFolderWithPath(_tv_datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target, path
    _std_train_ds = _ImageFolderWithPath(_train_path, transform=_train_tf)
    _std_val_ds   = _ImageFolderWithPath(_val_path,   transform=_test_tf)
    _test_root    = _test_path if _test_path else _val_path
    _std_test_ds  = _ImageFolderWithPath(_test_root,  transform=_test_tf)

    if getattr(args, 'weight_sampler', False):
        _targets = [s[1] for s in _std_train_ds.samples]
        _cnt  = np.bincount(_targets, minlength=2)
        _sw   = (1.0 / np.maximum(_cnt, 1))[_targets]
        _samp = torch.utils.data.WeightedRandomSampler(
                    _sw.tolist(), num_samples=len(_targets), replacement=True)
        _std_train_loader = DataLoader(
            _std_train_ds, batch_size=args.batch_size, sampler=_samp,
            num_workers=args.num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=_dl_generator)
    else:
        _std_train_loader = DataLoader(
            _std_train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=_dl_generator)
    _std_val_loader  = DataLoader(
        _std_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=_seed_worker, generator=_dl_generator)
    single_center_loader = DataLoader(
        _std_test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=_seed_worker, generator=_dl_generator)
    print(f"[standard mode] train={len(_std_train_ds)}  val={len(_std_val_ds)}  "
          f"test={len(_std_test_ds)}  classes={_std_train_ds.classes}")
    
    if args.proposed == True:
        model = LENet(args=args)
    else:
        model = LENet_base(args=args)


    _YEL = "\033[93m"; _RED = "\033[91m"; _RST = "\033[0m"
    _ckpt_self = 'saved_model/LENet-serial-sparse_token-4experts-top_1-linear-q_former_depths1_1_1_1-num_query_200-query_dim_384.pth'
    if os.path.isfile(_ckpt_self):
        _ckpt_data = torch.load(_ckpt_self, map_location='cpu', weights_only=False)
        model.load_state_dict(_ckpt_data['state_dict'], strict=False)
        print(f"{_YEL}[pretrain] self ckpt loaded: {_ckpt_self}{_RST}")
    elif getattr(args, 'proposed', True) or getattr(args, 'imagenet_pretrain', False):
        try:
            import torch as _torch_pt
            _URLS = {
                "swin":     "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
                "convnext": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
            }
            def _load_bb(dst, url, tag):
                if dst is None:
                    print(f"{_RED}[pretrain] backbone 없음 (skip {tag}){_RST}"); return
                ck = _torch_pt.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)
                ref = ck.get("model", ck) if isinstance(ck, dict) else ck
                cur = dst.state_dict()
                match = {k: v for k, v in ref.items() if k in cur and v.shape == cur[k].shape}
                dst.load_state_dict(match, strict=False)
                print(f"{_YEL}[pretrain] {tag} matched {len(match)}/{len(cur)} keys (ref={len(ref)}){_RST}")
            _load_bb(getattr(model, "swintransformer", None), _URLS["swin"],     "swin_small(MS)")
            _load_bb(getattr(model, "convnext",        None), _URLS["convnext"], "convnext_small(FB)")
        except Exception as _pe:
            print(f"{_RED}[pretrain] 로딩 실패 — scratch: {_pe}{_RST}")
    else:
        print(f"{_RED}[pretrain] proposed=False + no saved_model ckpt — random init{_RST}")

    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(param)
    model.to(rank)
    metric_collection = MetricCollection([
        Accuracy(task=args.categories, num_classes=args.num_classes, average='macro'),
        Precision(task=args.categories, num_classes=args.num_classes, average='macro'),
        Recall(task=args.categories, num_classes=args.num_classes, average='macro'),
        Specificity(task=args.categories, num_classes=args.num_classes, average='macro'),
        F1Score(task=args.categories, num_classes=args.num_classes, average='macro'),
        AUROC(task=args.categories, num_classes=args.num_classes, average='macro',),
    ])
    _ckpt = 'saved_model/LENet-serial-sparse_token-4experts-top_1-linear-q_former_depths1_1_1_1-num_query_200-query_dim_384.pth'
    if os.path.isfile(_ckpt):
        checkpoint = torch.load(_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"{_YEL}[HSQ] pretrained weights loaded: {_ckpt}{_RST}")
    else:
        print(f"{_RED}[HSQ] pretrained checkpoint not found — random init: {_ckpt}{_RST}")
    args.output_csv=f'{args.model}_multi'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)
    log_path = os.path.join(args.log_dir, '{}_training.log'.format(args.model))
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    if k_fold == 0:
        logger.addHandler(file_handler)
        logging.info("Model Name:{}".format(args.model))
        logging.info("Model Configuration:\n{}".format(model))
        for attribute, value in vars(args).items():
            logger.info(f"{attribute}: {value}")
        logging.info("param:\n{}".format(param))
    if args.model_ema:
        model_ema = ModelEmaV3(model,decay=args.model_ema_decay,
                use_warmup=args.model_ema_warmup,
                device='cpu' if args.model_ema_force_cpu else None,
            )
    else:
        model_ema=None
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    if args.mixup is True:
        mixup_fn=Mixup(mixup_alpha=0.8, cutmix_alpha=1., prob=1.,switch_prob=0.5, num_classes=args.num_classes)
    else:
        mixup_fn = None
    best_test_auc = 0.
    best_val_auc = 0.
    start_epoch = 0
    num_updates =0
    _no_improve = 0  # early stopping counter
    metric_collection.to('cpu')

    for param in model.parameters():
        param.requires_grad = True

    len_train_loader = len(_std_train_loader)
    swin_params = list(model.swintransformer.parameters())
    convnext_params = list(model.convnext.parameters())
    other_params= [p for n,p in model.named_parameters()
                        if 'swintransformer' not in n and 'convnext' not in n]
    assert len(list(model.parameters())) == len(swin_params) + len(convnext_params) + len(other_params), "Learning rate configuration is incomplete."
    param_groups = [
        {'params': swin_params},
        {'params': convnext_params},
        {'params': other_params}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr*0.1, args.lr*0.1, args.lr*1.5],
                                    total_steps=args.epochs * len_train_loader, pct_start=0.05,
                                    div_factor=float('inf'), final_div_factor=1000)
    for epoch in range(start_epoch,args.epochs):
        train_loader = _std_train_loader
        val_loader   = _std_val_loader
        print(len(_std_train_ds), len(train_loader), 'trainset/loader length')
        if args.proposed == True:
            train_loss = train_one_epoch(model, optimizer, metric_collection,num_updates=num_updates,epoch=epoch,scheduler=scheduler,criterion_focal=criterion_focal,
                                                mixup_fn=mixup_fn,data_loader=train_loader, device=rank,ema_updata_epoch=args.ema_updata_epoch,
                                                criterion=criterion,scaler=scaler,aux_loss=args.aux_loss,model_ema=model_ema,adv_bpr=getattr(args,'adv_bpr',True),
                                                bpr_lambda=getattr(args,'bpr_lambda',0.3),
                                                bpr_warmup=getattr(args,'bpr_warmup',5))
        else:
            train_loss = train_one_epoch_base(model, optimizer, metric_collection,num_updates=num_updates,epoch=epoch,scheduler=scheduler,criterion_focal=criterion_focal,
                                                mixup_fn=mixup_fn,data_loader=train_loader, device=rank,ema_updata_epoch=args.ema_updata_epoch,
                                                criterion=criterion,scaler=scaler,aux_loss=args.aux_loss,model_ema=model_ema)
        train_metric=metric_info(metric_collection)
        logger.info(f"Train: loss:{train_loss:.4f}, {train_metric}")
        if epoch>args.ema_updata_epoch:
            if args.proposed == True:
                val_loss = evaluate(model=model, data_loader=val_loader, criterion=criterion,
                                    metric_collection=metric_collection,device=rank,args=args)
            else:
                val_loss = evaluate_base(model=model, data_loader=val_loader, criterion=criterion,
                                    metric_collection=metric_collection,device=rank,args=args)
            val_metric=metric_info(metric_collection)
            logger.info(f"Val: loss:{val_loss:.4f}, {val_metric}")
            if val_metric['AUROC'] > best_val_auc:
                best_val_auc = val_metric['AUROC']
                _no_improve = 0
                num_updates += 1
                model_ema.update(model, step=num_updates)
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(args.checkpoints,
                                                    args.model + '_val{}.pth'.format(k_fold)))
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model_ema.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(args.checkpoints,
                                                    args.model + '_ema{}.pth'.format(k_fold)))
            else:
                _no_improve += 1
                _patience = getattr(args, 'patience', 0)
                if _patience > 0 and _no_improve >= _patience:
                    logger.info(f"Early stopping at epoch {epoch} (no improve for {_no_improve} epochs, best AUC={best_val_auc:.4f})")
                    print(f"[HSQ] Early stopping at epoch {epoch}")
                    break
        else:
            if args.proposed == True:
                val_loss = evaluate(model=model, data_loader=val_loader, criterion=criterion,
                                    metric_collection=metric_collection,device=rank,args=args)
            else:
                val_loss = evaluate_base(model=model, data_loader=val_loader, criterion=criterion,
                                    metric_collection=metric_collection,device=rank,args=args)
            val_metric = metric_info(metric_collection)
            logger.info(f"Val: loss:{val_loss:.4f}, {val_metric}")
            if args.proposed == True:
                test_loss = evaluate(model=model, data_loader=single_center_loader, criterion=criterion,
                                metric_collection=metric_collection,device=rank,args=args)
            else:
                test_loss = evaluate_base(model=model, data_loader=single_center_loader, criterion=criterion,
                                metric_collection=metric_collection,device=rank,args=args)
            test_metric=metric_info(metric_collection)
            logger.info(f"Single_center_test: loss:{test_loss:.4f}, {test_metric}")
            if val_metric['AUROC'] > best_val_auc:
                best_val_auc = val_metric['AUROC']
                _no_improve = 0
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(args.checkpoints,
                                                args.model + '_val{}.pth'.format(k_fold)))
            else:
                _no_improve += 1
                _patience = getattr(args, 'patience', 0)
                if _patience > 0 and _no_improve >= _patience:
                    logger.info(f"Early stopping at epoch {epoch} (no improve for {_no_improve} epochs, best AUC={best_val_auc:.4f})")
                    print(f"[HSQ] Early stopping at epoch {epoch}")
                    break
    #####
    if args.proposed == True:
        test_loss = evaluate(model=model, data_loader=single_center_loader, criterion=criterion,
                        metric_collection=metric_collection,device=rank,args=args)
    else:
        test_loss = evaluate_base(model=model, data_loader=single_center_loader, criterion=criterion,
                        metric_collection=metric_collection,device=rank,args=args)
    test_accuracy, test_precision, test_recall, test_specificity, test_f1, test_AUC = metric_collection.compute().values()
    print(
    "test_loss:{:.4f} test_accuracy:{:.4f} test_precision:{:.4f} test_recall:{:.4f} test_specificity:{:.4f} test_f1:{:.4f} test_AUC:{:.4f}" \
        .format(test_loss, test_accuracy, test_precision, test_recall,test_specificity, test_f1, test_AUC))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--mixup', default=False,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--categories', type=str, default='binary', choices=['binary', 'multiclass', 'multilabel'])
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', default='AdamW', help='optimizer')
    parser.add_argument('--seed', type=int, default=42, help='set seed')
    # --k_fold_index N : run only fold N (called per-fold from train.sh).
    # Default -1 = run all 5 folds in one process (legacy standalone mode).
    parser.add_argument('--k_fold_index', type=int, default=-1,
                        help='Run only this fold index (-1 = all folds)')
    # Training mode
    parser.add_argument('--mode', type=str, default='episode',
                        choices=['episode', 'standard'],
                        help='episode = HSQ 자체 npy 에피소드, standard = imagefolder')
    parser.add_argument('--train-path', dest='train_path', type=str, default=None,
                        help='[standard mode] train imagefolder path')
    parser.add_argument('--val-path',   dest='val_path',   type=str, default=None,
                        help='[standard mode] val imagefolder path')
    parser.add_argument('--test-path',  dest='test_path',  type=str, default=None,
                        help='[standard mode] test imagefolder path (optional; defaults to val)')
    parser.add_argument('--aux_loss', default=False, help='aux_loss')
    parser.add_argument('--resume', type=bool, default=False, help='put the path to resuming file if needed')
    parser.add_argument('--criterion', default='CrossEntropyLoss', help='criterion')
    parser.add_argument('--lr_param_groups', default=True, help='lr_param_groups')
    parser.add_argument('--weight_sampler', default=False, help='lr_param_groups')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                       help='Decay factor for model weights moving average (default: 0.9998)')
    parser.add_argument('--model-ema-warmup', action='store_true',
                       help='Enable warmup for model EMA decay.')
    parser.add_argument('--ema_updata_epoch', type=int, default=4)
    parser.add_argument('--model_ema', type=bool, default=True) 
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--amp', default=True, help='Automatic Mixed Precision')
    parser.add_argument('--distributed', default=False, help='distributed')
    parser.add_argument('--cam_visualization', type=bool, default=False)
    parser.add_argument('--visual_feature', type=bool, default=False)
    parser.add_argument('--k_fold', type=bool, default=True)
    parser.add_argument('--statistics', type=bool, default=False)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    #ModelArguments
    parser.add_argument('--dataset', type=str, default='liver')
    parser.add_argument('--model_name', type=str, default='hsq')
    parser.add_argument('--proposed', action='store_true')
    parser.add_argument('--model', type=str, default='LENet')
    parser.add_argument('--serial_parallel', type=str, default='serial',
                        choices=['parallel', 'serial','invert_serial'],help='Types of attention')
    parser.add_argument('--sparse_dense', type=str, default='sparse_token',
                        choices=['sparse_token', 'dense_token','sparse_expert','dense_expert','mlp'],
                        help='Types of MOE')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--head_type', type=str, default='linear',
                        choices=['moe_head', 'linear','mlp'],
                        )
    parser.add_argument('--cat_moe_head', type=bool, default=False)
    parser.add_argument('--q_former_depths', type=list, default=[1,1,1,1])
    parser.add_argument('--stage_dims', type=list, default=[96,192,384,768])
    parser.add_argument('--q_former_head_num', type=list, default=[3,6,12,24])
    parser.add_argument('--num_query_tokens', type=int, default=200)
    parser.add_argument('--query_dim', type=int, default=384)

    args = parser.parse_args()

    if args.k_fold_index >= 0:
        # Per-fold mode: called from train.sh with SEED env var already set
        print('Fold {}'.format(args.k_fold_index))
        main(0, args.k_fold_index, args)
    elif args.k_fold:
        # Legacy all-in-one mode: iterate all 5 folds in one process
        k_fold = 5
        for i in range(0, k_fold):
            print('第{}折'.format(i))
            main(0, i, args)
    else:
        main(0, 0, args)

