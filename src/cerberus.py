import torch
from torch import nn

class MatFact(nn.Module):
    """ Matrix factorization + user & item bias, weight init., sigmoid_range """

    def __init__(self, N, M, D, K, T, lambda_1=1, lambda_2=1):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(T, N, K))
        self.W = nn.Parameter(torch.zeros(T, D, K))
        self.V = nn.Parameter(torch.zeros(T, M, K))

        self.U_bias = nn.Parameter(torch.zeros(T, N))
        self.V_bias = nn.Parameter(torch.zeros(T, M))
        self.W_bias = nn.Parameter(torch.zeros(T, D))

        self.offset = nn.Parameter(torch.zeros(T, 1))

        self.U.data.uniform_(0., 0.05)
        self.W.data.uniform_(0., 0.05)
        self.V.data.uniform_(0., 0.05)

    def l2(self, x):
        return torch.sum(torch.pow(x, 2))

    def weight_regularization(self):
        V_reg = self.l2(self.V)
        W_reg = self.l2(self.W)
        U_reg = self.l2(self.U)
        return V_reg + W_reg + U_reg

    def alignment_regularization(self):
        V_reg = self.l2(self.V[1:] - self.V[:-1])
        W_reg = self.l2(self.W[1:] - self.W[:-1])
        U_reg = self.l2(self.U[1:] - self.U[:-1])

        return V_reg + W_reg + U_reg

    def get_cx(self, t, source):
        if source == 'A':
            cx_emb = self.V[t]
            cx_bias = self.V_bias[t]
        elif source == 'C':
            cx_emb = self.W[t]
            cx_bias = self.W_bias[t]
        else:
            raise ValueError(f"Invalid source: {source}. Expected 'A' or 'C'.")

        return cx_emb, cx_bias

    def forward(self, t, user_ind, source):
        cx_emb, cx_bias = self.get_cx(t, source[0])

        user_emb = self.U[t, user_ind].unsqueeze(1)
        user_bias = self.U_bias[t, user_ind].unsqueeze(1)

        user_emb = torch.permute(user_emb, [0, 2, 1])
        element_product = torch.matmul(cx_emb, user_emb).squeeze()
        element_product += user_bias + cx_bias + self.offset[t]

        return element_product