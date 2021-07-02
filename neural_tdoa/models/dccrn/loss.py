import torch
import torch.nn.functional as F


def loss(self, inputs, labels, loss_mode='SI-SNR'):

    if loss_mode == 'MSE':
        _, d, _ = inputs.shape
        labels[:, 0, :] = 0
        labels[:, d//2, :] = 0
        return F.mse_loss(inputs, labels, reduction='mean')*d

    elif loss_mode == 'SI-SNR':
        return -(si_snr(inputs, labels))
    elif loss_mode == 'MAE':
        gth_spec, _ = self.stft(labels)
        _, d, _ = inputs.shape
        return torch.mean(torch.abs(inputs-gth_spec))*d


def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)
