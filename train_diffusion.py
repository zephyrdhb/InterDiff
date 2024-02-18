import argparse
import os
import shutil
import numpy as np
import torch
# import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from tqdm.auto import tqdm
from add_finetune import BottleneckModel
import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_processed_data
import torch.nn.functional as F
from models.new_score_model2 import ScorePosNet3D, to_torch_const,log_1_min_a
from collections import OrderedDict as od
import lmdb,pickle
from torch.utils.data import Subset
from pathlib import Path
def cosine_beta_schedule(timesteps, power = 1,s=0.01):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps)**power + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas
def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)

used_keys = ['protein_pos','ligand_pos','protein_atom_feature','ligand_atom_feature_full','protein_element','ligand_element']
def index_to_log_onehot(x, num_classes=15):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def pad_sequence(x,max_length):
    if len(x.shape) == 1:
        x = index_to_log_onehot(x)
    if x.shape[0] < max_length:
        new_x = x.new_zeros([1,max_length,x.shape[1]])
        new_x[:,:x.shape[0],:] = x
        return new_x
    else:
        return x.unsqueeze(0)

def collate_fun(batch):
    # max_poc = max(i['protein_element'].shape[0] for i in batch)
    # max_lig = max(i['ligand_element'].shape[0] for i in batch)
    data = {}
    data['interaction'] = torch.cat([j['interaction'] for j in batch]).squeeze().long()
    for i in used_keys:
        if 'element' in i:
            data[i+'_batch'] = torch.tensor(sum([[idx]* len(j[i]) for idx,j in enumerate(batch)],[]))
            data[i+'_length'] = [len(j[i]) for j in batch]
        else:
            data[i] = torch.cat([torch.tensor(j[i]).clone().detach() for j in batch])
    return data

def mark_embedding_as_fix(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "prompt" in n or 'node_mlp' in n:
            p.requires_grad = False

def zero_first_embedding(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "prompt" in n:
            nn.init.constant_(p.weight[0],0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--reschedule', type=bool, default=False)
    parser.add_argument('--train_report_iter', type=int, default=20)
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_processed_data()

    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    train_iterator = utils_train.inf_iterator(torch.utils.data.DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collate_fun
    ))

    val_loader = torch.utils.data.DataLoader(val_set, config.train.batch_size, shuffle=False,
    collate_fn = collate_fun)
    scaler = torch.cuda.amp.GradScaler()
    # Model
    logger.info('Building model...')

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=25,
        ligand_atom_feature_dim=12,
        use_four_emb = True,
        device = args.device
    ).to(args.device)

    model = BottleneckModel(model).to(args.device)
    ##
    # dict_buffer = torch.load('./logs_diffusion/training_2023_11_16__13_16_42/checkpoints/time_dep_5760.pt',map_location=lambda storage, loc:storage.cuda(1))
    # re_dict = od()
    # for k,v in dict_buffer['model'].items():
    #     re_dict[k[10:]] = v
    # model.load_state_dict(re_dict, strict=False)

    mark_embedding_as_fix(model)#fix the parameters of interactions embedding and classification part. This could be done to finetune the model

    if args.reschedule:
        alphas_v = cosine_beta_schedule(1000,2)
        log_alphas_v = np.log(alphas_v).squeeze()
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        model.state_dict()['model.log_alphas_v'].copy_(to_torch_const(log_alphas_v))
        model.state_dict()['model.log_one_minus_alphas_v'].copy_(to_torch_const(log_1_min_a(log_alphas_v)))
        model.state_dict()['model.log_alphas_cumprod_v'].copy_(to_torch_const(log_alphas_cumprod_v))
        model.state_dict()['model.log_one_minus_alphas_cumprod_v'].copy_(to_torch_const(log_1_min_a(log_alphas_cumprod_v)))
    # model = torch.compile(model, mode="max-autotune")

    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator)
            protein_noise = torch.randn_like(batch['protein_pos']) * config.train.pos_noise_std
            gt_protein_pos = batch['protein_pos'] + protein_noise
            with torch.cuda.amp.autocast():
                results = model.get_diffusion_loss(
                    protein_pos=gt_protein_pos.to(args.device),
                    protein_v=batch['protein_atom_feature'].float().to(args.device),
                    batch_protein=batch['protein_element_batch'].to(args.device),
                    protein_len=batch['protein_element_length'],

                    ligand_pos=batch['ligand_pos'].to(args.device),
                    ligand_v=batch['ligand_atom_feature_full'].to(args.device),
                    batch_ligand=batch['ligand_element_batch'].to(args.device),
                    ligand_len=batch['ligand_element_length'],
                    interaction = batch['interaction']
                )
                loss, loss_pos, loss_v,loss_cls = results['loss'], results['loss_pos'], results['loss_v'], results['loss_cls']
                loss = loss / config.train.n_acc_batch
                scaler.scale(loss).backward()
        orig_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), config.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | cls %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v,loss_cls, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )

    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_cls,sum_n = 0, 0, 0, 0, 0
        all_pred_v, all_true_v = [], []

        with torch.no_grad():
            model.eval()
            torch.cuda.empty_cache()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch
                batch_size = 10
                
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)

                    results = model.get_diffusion_loss(
                    protein_pos=batch['protein_pos'].to(args.device),
                    protein_v=batch['protein_atom_feature'].float().to(args.device),
                    batch_protein=batch['protein_element_batch'].to(args.device),
                    protein_len=batch['protein_element_length'],

                    ligand_pos=batch['ligand_pos'].to(args.device),
                    ligand_v=batch['ligand_atom_feature_full'].to(args.device),
                    batch_ligand=batch['ligand_element_batch'].to(args.device),
                    ligand_len=batch['ligand_element_length'],
                    interaction = batch['interaction'],
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v, loss_cls = results['loss'], results['loss_pos'], results['loss_v'], results['loss_cls']

                    sum_loss += float(loss)
                    sum_loss_pos += float(loss_pos)
                    sum_loss_v += float(loss_v)
                    sum_loss_cls += float(loss_cls)
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch['ligand_atom_feature_full'].detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_cls = sum_loss_cls / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode='add_aromatic')
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss cls %.6f | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000,avg_loss_cls, atom_auroc
            )
        )
        # writer.add_scalar('val/loss', avg_loss, it)
        # writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        # writer.add_scalar('val/loss_v', avg_loss_v, it)
        # writer.add_scalar('val/loss_cls', avg_loss_cls, it)
        # writer.flush()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        for it in tqdm(range(1, config.train.max_iters + 1)):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, 'time_dep_%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')