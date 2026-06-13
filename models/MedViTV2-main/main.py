import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import requests
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from datasets import build_dataset
from distutils.util import strtobool
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import natten
def _natten_determinism():
    applied, skipped = [], []
    _knobs = [
        ("kv_parallelism_off", lambda: natten.use_kv_parallelism_in_fused_na(False)),
        ("mem_strict",         lambda: natten.set_memory_usage_preference("strict")),
        ("autotuner_off",      lambda: natten.use_autotuner(False, False, False, False)),
        ("deterministic_on",   lambda: natten.use_deterministic_algorithms(True)),
    ]
    for name, fn in _knobs:
        try:
            fn(); applied.append(name)
        except Exception:
            skipped.append(name)
    try:
        if hasattr(natten, "context"):
            ctx = natten.context
            for name, fn in [
                ("ctx_kv_off",      lambda: ctx.set_kv_parallelism_in_fused_na(False)),
                ("ctx_mem_strict",  lambda: ctx.set_memory_usage_preference("strict")),
            ]:
                try:
                    fn(); applied.append(name)
                except Exception:
                    skipped.append(name)
    except Exception:
        pass
    print(f"[natten] determinism applied={applied} skipped={skipped} "
          f"(natten {getattr(natten, '__version__', '?')})")

_natten_determinism()
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
#from MedViTV1 import MedViT_small, MedViT_base, MedViT_large


model_classes = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large
}

model_urls = {
    "MedViT_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViT_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0"
}

def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")

def train_mnist(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, data_flag, task):
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            
            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels)
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs.squeeze(0), labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        net.eval()
        y_score = torch.tensor([])
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                inputs, targets = val_data
                outputs = net(inputs.to(device))
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                
                y_score = torch.cat((y_score, outputs.cpu()), 0)
                
        y_score = y_score.detach().numpy()
        evaluator = Evaluator(data_flag, 'test', size=224, root='./data')
        metrics = evaluator.evaluate(y_score)
        
        val_accurate, _ = metrics
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}')
        #print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')

def specificity_per_class(conf_matrix):
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))
    return specificity

def overall_accuracy(conf_matrix):
    tp_tn_sum = conf_matrix.trace()
    total_sum = conf_matrix.sum()
    return tp_tn_sum / total_sum

def train_other(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path):
    best_acc = 0.0
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        acc = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                probs = torch.softmax(outputs, dim=1)

                predict_y = torch.max(probs, dim=1)[1]

                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        val_accurate = acc / len(test_loader.dataset)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = specificity_per_class(conf_matrix)
        avg_specificity = sum(specificity) / len(specificity)

        overall_acc = overall_accuracy(conf_matrix)

        n_classes = len(conf_matrix)
        all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))

        try:
            auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
        except ValueError:
            auc = float('nan')

        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} '
              f'val_accuracy: {val_accurate:.4f} precision: {precision:.4f} '
              f'recall: {recall:.4f} specificity: {avg_specificity:.4f} '
              f'f1_score: {f1:.4f} auc: {auc:.4f} overall_accuracy: {overall_acc:.4f}')
        
        #print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')

def train_other_with_early_stop(epochs, net, train_loader, test_loader, optimizer, scheduler,
                                loss_function, device, save_path, patience=20):
    BEST_BY = "auc"
    print(f"[train_other] best ckpt by {BEST_BY.upper()}")

    best_score = 0.0
    wait = 0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        net.eval()
        ys_list, scores_list = [], []
        with torch.no_grad():
            for val_data in test_loader:
                vi, vl = val_data
                out = net(vi.to(device))
                prob = torch.softmax(out, dim=1)
                ys_list.append(vl.cpu().numpy().astype(int).ravel())
                if prob.shape[1] == 2:
                    scores_list.append(prob[:, 1].cpu().numpy())
                else:
                    scores_list.append(prob.cpu().numpy())
        ys = np.concatenate(ys_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        if scores.ndim == 1:
            preds = (scores >= 0.5).astype(int)
        else:
            preds = scores.argmax(axis=1)
        val_acc = float((preds == ys).mean())
        try:
            if scores.ndim == 1:
                val_auc = float(roc_auc_score(ys, scores)) if len(set(ys.tolist())) > 1 else 0.0
            else:
                val_auc = float(roc_auc_score(ys, scores, multi_class='ovr'))
        except Exception:
            val_auc = 0.0
        val_f1 = float(f1_score(ys, preds, average='macro'))
        val_score = {"acc": val_acc, "auc": val_auc, "f1": val_f1}[BEST_BY]
        print(f"[epoch {epoch+1}] acc={val_acc:.4f}  auc={val_auc:.4f}  f1={val_f1:.4f}  "
              f"best[{BEST_BY}]={best_score:.4f}  wait={wait}/{patience}")

        if val_score > best_score + 1e-6:
            best_score = val_score
            wait = 0
            torch.save(net.state_dict(), save_path)
            print(f"  -> new best [{BEST_BY}]={best_score:.4f}, saved to {save_path}")
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"[early-stop] no improvement {wait} epochs. Stopping.")
                break
    print(f"Best val_{BEST_BY}: {best_score:.4f}")


def main(args):
    seed = int(getattr(args, 'seed', 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    _dl_generator = torch.Generator()
    _dl_generator.manual_seed(seed)
    print(f"[seed] fixed at {seed} (torch+cuda+np+random, cudnn deterministic)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    model_name = args.model_name
    dataset_name = args.dataset
    pretrained = args.pretrained
    if args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        if task == "multi-label, binary-class":
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    model_class = model_classes.get(model_name)

    # if not model_class:
    #     raise ValueError(f"Model {model_name} is not recognized. Available models: {list(model_classes.keys())}")

    batch_size = args.batch_size
    lr = args.lr
    
    train_dataset, test_dataset, nb_classes = build_dataset(args=args)
    val_num = len(test_dataset)
    train_num = len(train_dataset)
    
    eta = args.epochs * train_num // args.batch_size

    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=nb_classes).cuda()
        if pretrained:
            checkpoint_path = args.checkpoint_path
            if not os.path.exists(checkpoint_path):
                _local = f'./{model_name}.pth'
                if os.path.exists(_local):
                    print(f"[pretrained] using local checkpoint: {_local}")
                    checkpoint_path = _local
                else:
                    checkpoint_url = model_urls.get(model_name)
                    if not checkpoint_url:
                        raise ValueError(f"Checkpoint URL for model {model_name} not found.")
                    print(f"[pretrained] local {_local} not found — downloading from {checkpoint_url[:80]}...")
                    try:
                        download_checkpoint(checkpoint_url, _local)
                        checkpoint_path = _local
                    except Exception as _e:
                        raise RuntimeError(
                            f"[pretrained] download failed: {_e}\n"
                            f"   -> place the pretrained file at '{_local}' or\n"
                            f"      specify --checkpoint_path explicitly."
                        )

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = net.state_dict()
            for k in ['proj_head.0.weight', 'proj_head.0.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            net.load_state_dict(checkpoint, strict=False)
    else:
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=nb_classes).cuda()

    
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eta, eta_min=5e-6)
    
    if getattr(args, 'weighted_sampler', False) and hasattr(train_dataset, 'targets'):
        import numpy as _np
        _targets = train_dataset.targets if hasattr(train_dataset, 'targets') else \
                   [int(s[1]) for s in getattr(train_dataset, 'samples', [])]
        _cnt = _np.bincount(_targets, minlength=nb_classes)
        _w = 1.0 / _np.maximum(_cnt, 1)
        _sw = _w[_targets]
        _sampler = data.WeightedRandomSampler(_sw.tolist(), num_samples=len(_targets), replacement=True)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=_sampler, worker_init_fn=_seed_worker, generator=_dl_generator)
        print(f'[weighted-sampler] class_count={_cnt.tolist()}  class_weight={_w.tolist()}')
    else:
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_seed_worker, generator=_dl_generator)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False, worker_init_fn=_seed_worker, generator=_dl_generator)
    
    print(train_dataset)
    print("===================")
    print(test_dataset)

    epochs = args.epochs
    best_acc = 0.0
    save_path = f'./{model_name}_{dataset_name}.pth'
    train_steps = len(train_loader)

    _out_dir = getattr(args, 'output_dir', './') or './'
    os.makedirs(_out_dir, exist_ok=True)
    save_path = os.path.join(_out_dir, f'{model_name}_{dataset_name}_best.pth')

    def _eval_and_save_npz(net, loader, device, out_npz):
        import numpy as _np
        net.eval()
        ys, ps = [], []
        with torch.no_grad():
            for vb in loader:
                vimg, vlbl = vb[0], vb[1]
                if hasattr(vlbl, 'squeeze'):
                    vlbl = vlbl.squeeze().long() if vlbl.dim() > 1 else vlbl
                out = net(vimg.to(device))
                prob = torch.softmax(out, dim=1)
                ps.append(prob.cpu().numpy())
                ys.append(vlbl.cpu().numpy())
        y_true = _np.concatenate(ys, axis=0).astype(int)
        y_score = _np.concatenate(ps, axis=0)
        if y_score.shape[1] > 2:
            pass
        os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or '.', exist_ok=True)
        _np.savez(out_npz, y_true=y_true, y_score=y_score)
        print(f'[save-predictions] wrote: {out_npz}  N={len(y_true)}')

    def _unwrap_ckpt(_sd):
        if isinstance(_sd, dict):
            for k in ('model', 'state_dict', 'net'):
                if k in _sd and isinstance(_sd[k], dict):
                    return _sd[k]
        return _sd

    def _load_and_report(_sd_raw, tag):
        _sd = _unwrap_ckpt(_sd_raw)
        missing, unexpected = net.load_state_dict(_sd, strict=False)
        n_match = sum(1 for k in _sd.keys() if k in net.state_dict())
        n_total = len(_sd) if hasattr(_sd, '__len__') else 0
        print(f'[{tag}] loaded ckpt: matched={n_match}/{n_total}  '
              f'missing={len(missing)}  unexpected={len(unexpected)}')
        if n_match == 0:
            print(f'[{tag}] WARNING: no weights matched — check ckpt format')

    if getattr(args, 'eval', False):
        _ckpt = args.checkpoint_path
        if os.path.isfile(_ckpt):
            _sd_raw = torch.load(_ckpt, map_location='cpu', weights_only=False)
            _load_and_report(_sd_raw, 'eval')
        if args.save_predictions:
            _eval_and_save_npz(net, test_loader, device, args.save_predictions)
        return

    if dataset_name.endswith('mnist'):
        train_mnist(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path, dataset_name, task)
    else:
        train_other_with_early_stop(epochs, net, train_loader, test_loader,
            optimizer, scheduler, loss_function, device, save_path,
            patience=getattr(args, 'early_stop_patience', 20))

    if args.save_predictions and os.path.isfile(save_path):
        _best_raw = torch.load(save_path, map_location='cpu', weights_only=False)
        _load_and_report(_best_raw, 'post-train-eval')
        _eval_and_save_npz(net, test_loader, device, args.save_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for MedViT models.')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny', help='Model name to use.')
    #tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, breastmnist, bloodmnist,
    #organamnist, organcmnist, organsmnist'
    parser.add_argument('--dataset', type=str, default='PAD', help='Dataset to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default=False, help="Whether to use pretrained weights (True/False).")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/MedViT_tiny.pth', help='Path to the checkpoint file.')
    parser.add_argument('--train-path', type=str, default=None)
    parser.add_argument('--val-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./')
    parser.add_argument('--save-predictions', type=str, default='')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--weighted-sampler', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()
    main(args)

# python main.py --model_name 'convnext_tiny' --dataset 'PAD'
