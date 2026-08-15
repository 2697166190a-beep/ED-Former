import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from util import AverageMeter, write_img, chw_to_hwc
from loader import PairLoader
from ED_Former import ED_Former
from fvcore.nn import FlopCountAnalysis, flop_count_table
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ed-former', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):

    checkpoint = torch.load(save_dir, map_location=torch.device('cpu'))

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] 
        else:
            name = k 
        new_state_dict[name] = v

    return new_state_dict

def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '%.02f | %.04f.csv' % (PSNR.avg, SSIM.avg)))


if __name__ == '__main__':

    MODEL_REGISTRY = {
        'ed-former': ED_Former,
    }
    model_name_lower = args.model.lower()
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"Model '{args.model}' not found in registry. "
                         f"Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name_lower]
    network = model_class()
    network.cuda()

    input_size = (1, 3, 256, 256)
    input_tensor = torch.randn(input_size).cuda()

    flops = FlopCountAnalysis(network, input_tensor)
    total_flops = flops.total()
    total_macs = total_flops / 2

    params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("               Model Efficiency Analysis")
    print("=" * 60)
    print(f"Model Name:         {args.model}")
    print(f"Input Shape:        {input_size}")
    print(f"Trainable Params:   {params / 1e6:.2f} M")
    print(f"FLOPs:              {total_flops / 1e9:.2f} G")
    print(f"MACs (estimated):   {total_macs / 1e9:.2f} G")
    print("-" * 60)

    print("=" * 60 + "\n")
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)

        checkpoint = torch.load(saved_model_dir, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

        network.load_state_dict(new_state_dict)
        print(f"==> The model weights were successfully loaded from {saved_model_dir}")

    else:
        print('==> No trained model was found!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    test(test_loader, network, result_dir)
