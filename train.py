import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import models
from util import AverageMeter
from loader import PairLoader
from ED_Former import ED_Former
import logging
from datetime import datetime
from HierarchicalInvarianceLoss import HI_Loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ed-former', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--lambda_l1_start', default=0.5, type=float, help='initial weight for L1 loss')
parser.add_argument('--lambda_l1_end', default=0.9, type=float, help='final weight for L1 loss')
args = parser.parse_args()

def train(train_loader, network, criterion, optimizer, scaler, epoch, total_epochs, device):
    total_losses = AverageMeter()
    l1_losses = AverageMeter()
    perceptual_losses = AverageMeter()
    lambda_tracker = AverageMeter()
    torch.cuda.empty_cache()
    network.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        source_img = batch['source'].to(device, non_blocking=True)
        target_img = batch['target'].to(device, non_blocking=True)

        with autocast(enabled=not args.no_autocast):
            output = network(source_img)
            total_loss, loss_components = criterion(output, target_img, epoch, total_epochs)

        total_losses.update(total_loss.item())
        l1_losses.update(loss_components['l1_loss'].item())
        perceptual_losses.update(loss_components['perceptual_loss'].item())
        lambda_tracker.update(loss_components['current_lambda_l1'])

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix(
            total_loss=f'{total_losses.avg:.4f}',
            l1=f'{l1_losses.avg:.4f}',
            percep=f'{perceptual_losses.avg:.4f}',
            l1_w=f'{lambda_tracker.val:.2f}'
        )

    return {
        'total_loss_avg': total_losses.avg,
        'l1_loss_avg': l1_losses.avg,
        'perceptual_loss_avg': perceptual_losses.avg,
        'lambda_l1_avg': lambda_tracker.avg 
    }


def valid(val_loader, network, device):
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    network.eval()
    for batch in val_loader:
        source_img = batch['source'].to(device, non_blocking=True)
        target_img = batch['target'].to(device, non_blocking=True)
        with torch.no_grad():
            output = network(source_img).clamp_(-1, 1)
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))
    return PSNR.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("===================================================")
        logging.info(f"CUDA is available. Forcing to use a single GPU.")
        logging.info(f"Target Device: {device}")
        logging.info(f"Device Name: {torch.cuda.get_device_name(device)}")
        logging.info("===================================================")
        num_gpus = 1
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Training will use CPU.")
        num_gpus = 0

    log_file_dir = os.path.join(args.log_dir, args.exp)
    os.makedirs(log_file_dir, exist_ok=True)
    current_time = datetime.now().strftime('%m%d')
    log_file_name = f"{args.model}_{current_time}.txt"
    log_file_path = os.path.join(log_file_dir, log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    MODEL_REGISTRY = {
        'ed-former': ED_Former,
    }

    model_name_lower = args.model.lower()
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"Model '{args.model}' not found in registry. "
                         f"Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name_lower]

    network = model_class()

    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    logging.info(f"Model name: {args.model}")
    logging.info(f"Total number of staff: {total_params / 1e6:.4f} M")
    logging.info(f"Number of trainable parameters: {trainable_params / 1e6:.4f} M")
    network.to(device)
    if num_gpus > 1:
        logging.info(f"Activating DataParallel for {num_gpus} GPUs.")
        network = nn.DataParallel(network)

    logging.info("The VGG16 model is being loaded for sensing loss...")
    vgg_model = models.vgg16(pretrained=True).features.to(device)
    vgg_model.eval()
    for param in vgg_model.parameters():
        param.requires_grad = False
    logging.info("The VGG16 model has been loaded and frozen.")

    criterion = HI_Loss(vgg_model=vgg_model, lambda_l1_start=args.lambda_l1_start,
                             lambda_l1_end=args.lambda_l1_end).to(device)

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset, batch_size=setting['batch_size'], shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], setting['patch_size'])
    val_loader = DataLoader(val_dataset, batch_size=setting['batch_size'], num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        logging.info('Start training. Current model name ' + args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0
        for epoch in range(1, setting['epochs'] + 1):
            loss_dict_avg = train(train_loader, network, criterion, optimizer, scaler, epoch, setting['epochs'], device)
            log_message = (
                f"Epoch: {epoch}/{setting['epochs']}, "
                f"Total Loss: {loss_dict_avg['total_loss_avg']:.4f}, "
                f"L1 Loss: {loss_dict_avg['l1_loss_avg']:.4f}, "
                f"Perceptual Loss: {loss_dict_avg['perceptual_loss_avg']:.4f}, "
                f"Best PSNR: {best_psnr:.4f}"
            )
            logging.info(log_message)

            writer.add_scalar('train_loss/total', loss_dict_avg['total_loss_avg'], epoch)
            writer.add_scalar('train_loss/l1', loss_dict_avg['l1_loss_avg'], epoch)
            writer.add_scalar('train_loss/perceptual', loss_dict_avg['perceptual_loss_avg'], epoch)
            writer.add_scalar('params/lambda_l1', loss_dict_avg['lambda_l1_avg'], epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network, device)
                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))
                writer.add_scalar('best_psnr', best_psnr, epoch)

    else:
        logging.info('There are existing training models')
        exit(1)
