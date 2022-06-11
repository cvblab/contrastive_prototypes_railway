from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_image(x, y=None, denorm_intensity=False, channel_first=True):
    if len(x.shape) < 3:
        x = np.expand_dims(x, 0)
    # channel first
    if channel_first:
        x = np.transpose(x, (1, 2, 0))
    if denorm_intensity:
        if self.norm == 'zscore':
            x = (x*127.5) + 127.5
            x = x.astype(int)

    plt.imshow(x)

    if y is not None:
        y = np.expand_dims(y[0, :, :], -1)
        plt.imshow(y, cmap='jet', alpha=0.1)

    plt.axis('off')
    plt.show()


def grad_cam(activations, output, normalization='relu_min_max', avg_grads=False, norm_grads=False):
    def normalize(grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2), (-1, -2, -3))) + 1e-5
        l2_norm = l2_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return grads * torch.pow(l2_norm, -1)

    # Obtain gradients
    gradients = torch.autograd.grad(output, activations, grad_outputs=None, retain_graph=True, create_graph=True,
                                    only_inputs=True, allow_unused=True)[0]

    # Normalize gradients
    if norm_grads:
        gradients = normalize(gradients)

    # pool the gradients across the channels
    if avg_grads:
        gradients = torch.mean(gradients, dim=[2, 3])
        gradients = gradients.unsqueeze(-1).unsqueeze(-1)

    # weight activation maps
    if 'relu' in normalization:
        GCAM = torch.sum(torch.relu(gradients * activations), 1)
    else:
        GCAM = torch.sum(gradients * activations, 1)

    # Normalize CAM
    if 'tanh' in normalization:
        GCAM = torch.tanh(GCAM)
    if 'sigm' in normalization:
        GCAM = torch.sigmoid(GCAM)
    if 'abs' in normalization:
        GCAM = torch.abs(GCAM)
    if 'min' in normalization:
        norm_value = torch.min(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM - norm_value
    if 'max' in normalization:
        norm_value = torch.max(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM * norm_value.pow(-1)
    if 'clamp' in normalization:
        GCAM = GCAM.clamp(max=1)

    return GCAM