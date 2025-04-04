import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible))
        self.a = nn.Parameter(torch.randn(1, n_hidden))  # Hidden bias
        self.b = nn.Parameter(torch.randn(1, n_visible))  # Visible bias

    def sample_h(self, v):
        activation = torch.mm(v, self.W.t()) + self.a
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        activation = torch.mm(h, self.W) + self.b
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk, lr=0.01):
        self.W.data += lr * (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b.data += lr * torch.sum((v0 - vk), 0)
        self.a.data += lr * torch.sum((ph0 - phk), 0)