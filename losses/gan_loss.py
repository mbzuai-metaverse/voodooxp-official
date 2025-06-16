# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md

# The code is modified from https://deepimagination.cc/_modules/imaginaire/losses/gan/
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def fuse_math_min_mean_pos(x):
    r"""Fuse operation min mean for hinge loss computation of positive
    samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


@torch.jit.script
def fuse_math_min_mean_neg(x):
    r"""Fuse operation min mean for hinge loss computation of negative
    samples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


class GANLoss(nn.Module):
    r"""GAN loss constructor.

    Args:
        gan_mode (str): Type of GAN loss. ``'hinge'``, ``'least_square'``,
            ``'non_saturated'``, ``'wasserstein'``.
        target_real_label (float): The desired output label for real images.
        target_fake_label (float): The desired output label for fake images.
        decay_k (float): The decay factor per epoch for top-k training.
        min_k (float): The minimum percentage of samples to select.
        separate_topk (bool): If ``True``, selects top-k for each sample
            separately, otherwise selects top-k among all samples.
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 decay_k=1., min_k=1., separate_topk=False):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.gan_mode = gan_mode

    def forward(self, dis_output, t_real, dis_update=True, reduce=True):
        r"""GAN loss computation.

        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the discriminator, otherwise the generator.
            reduce (bool): If ``True``, when a list of discriminator outputs are provided, it will return the average
                of all losses, otherwise it will return a list of losses.
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(dis_output, list):
            # For multi-scale discriminators.
            # In this implementation, the loss is first averaged for each scale
            # (batch size and number of locations) then averaged across scales,
            # so that the gradient is not dominated by the discriminator that
            # has the most output values (highest resolution).
            losses = []
            for dis_output_i in dis_output:
                assert isinstance(dis_output_i[-1], torch.Tensor)
                assert isinstance(dis_output_i, list)
                losses.append(self.loss(dis_output_i[-1], t_real, dis_update))
            if reduce:
                return torch.mean(torch.stack(losses))
            else:
                return losses
        else:
            return self.loss(dis_output, t_real, dis_update)

    def loss(self, dis_output, t_real, dis_update=True):
        r"""GAN loss computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if self.gan_mode == 'non_saturated':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = F.binary_cross_entropy_with_logits(dis_output,
                                                      target_tensor)
        elif self.gan_mode == 'least_square':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = 0.5 * F.mse_loss(dis_output, target_tensor)
        elif self.gan_mode == 'hinge':
            if dis_update:
                if t_real:
                    loss = fuse_math_min_mean_pos(dis_output)
                else:
                    loss = fuse_math_min_mean_neg(dis_output)
            else:
                loss = -torch.mean(dis_output)
        elif self.gan_mode == 'wasserstein':
            if t_real:
                loss = -torch.mean(dis_output)
            else:
                loss = torch.mean(dis_output)
        elif self.gan_mode == 'softplus':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = F.binary_cross_entropy_with_logits(dis_output,
                                                      target_tensor)
        else:
            raise ValueError('Unexpected gan_mode {}'.format(self.gan_mode))
        return loss

    def get_target_tensor(self, dis_output, t_real):
        r"""Return the target vector for the binary cross entropy loss
        computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
        Returns:
            target (tensor): Target tensor vector.
        """
        if t_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = dis_output.new_tensor(self.real_label)
            return self.real_label_tensor.expand_as(dis_output)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = dis_output.new_tensor(self.fake_label)
            return self.fake_label_tensor.expand_as(dis_output)

    def r1_penalty(self, D, real_img, condition):
        real_img_tmp = real_img.detach().requires_grad_(True)
        real_logits = D(real_img_tmp, condition)
        r1_penalty = 0

        if type(real_logits) is not list:
            real_logits = [[real_logits]]

        for real_logit in real_logits:
            r1_grads = torch.autograd.grad(
                outputs=[real_logit[-1].sum()], inputs=[real_img_tmp], create_graph=True
            )[0]
            r1_penalty += r1_grads.square().sum([1, 2, 3])
        r1_penalty = r1_penalty.mean()

        return r1_penalty


class FeatMatchingLoss(nn.Module):
    def __init__(self, n_layers_D, num_D):
        super().__init__()

        self.criterionFeat = torch.nn.L1Loss()
        self.n_layers_D = n_layers_D
        self.num_D = num_D

    def forward(self, pred_fake, pred_real):
        loss = 0

        feat_weights = 4.0 / (self.n_layers_D + 1)
        D_weights = 1.0 / self.num_D

        for i in range(self.num_D):
            for j in range(len(pred_fake[i]) - 1):
                feat_dist = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                loss += D_weights * feat_weights * feat_dist

        return loss
