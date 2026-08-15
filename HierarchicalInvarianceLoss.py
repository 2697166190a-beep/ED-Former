import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):

        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):

        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):

        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))
        return sum(loss) / len(loss)



class HI_Loss(nn.Module):
    def __init__(self, vgg_model, lambda_l1_start=0.7, lambda_l1_end=0.9):

        super(HI_Loss, self).__init__()
        self.lambda_l1_start = lambda_l1_start
        self.lambda_l1_end = lambda_l1_end

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LossNetwork(vgg_model)

        print("The dynamic weight combination loss function has been created：")
        print(f"  - L1 Weight range: {self.lambda_l1_start} -> {self.lambda_l1_end}")

    def forward(self, dehaze, gt, current_epoch, total_epochs):

        progress = current_epoch / total_epochs
        cos_annealing = (1 + math.cos(math.pi * progress)) / 2
        lambda_l1 = self.lambda_l1_end - (self.lambda_l1_end - self.lambda_l1_start) * cos_annealing
        lambda_perceptual = 1.0 - lambda_l1

        loss_l1 = self.l1_loss(dehaze, gt)

        loss_perceptual = self.perceptual_loss(dehaze, gt)

        total_loss = (lambda_l1 * loss_l1) + (lambda_perceptual * loss_perceptual)

        loss_components = {
            'total_loss': total_loss,
            'l1_loss': loss_l1,
            'perceptual_loss': loss_perceptual,
            'current_lambda_l1': lambda_l1
        }

        return total_loss, loss_components
