import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np


class CNNScalarForm(nn.Module):

    def __init__(self, check_with_pytorch=False):
        super(CNNScalarForm, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.fc1 = nn.Linear(12 * 12 * 20, 500)
        self.fc2 = nn.Linear(500, 10)

        self.check_with_pytorch = check_with_pytorch

    def forward(self, x):
        if self.check_with_pytorch:
            return self._forward_with_cmp(x)

        return self._forward(x)


    def _forward(self, x):
        z_conv = self._conv(x, self.conv1.weight, self.conv1.bias)
        z_pool = self._max_pool_kernel2(z_conv)

        z_reshape = self._reshape(z_pool)

        z_fc1 = self._fc(z_reshape, self.fc1.weight, self.fc1.bias)

        z_relu = self._relu(z_fc1)

        z_fc2 = self._fc(z_relu, self.fc2.weight, self.fc2.bias)

        z_softmax = z_fc2.softmax(dim=1)

        return z_softmax

    def _forward_with_cmp(self, x):

        z_conv = self._conv(x, self.conv1.weight, self.conv1.bias)
        z_conv_target = self.conv1(x)
        mse = F.mse_loss(z_conv, z_conv_target)
        print(f"Check conv. MSE: {mse}")

        z_pool = self._max_pool_kernel2(z_conv)
        z_pool_target = F.max_pool2d(z_conv_target, 2, 2)
        mse = F.mse_loss(z_pool, z_pool_target)
        print(f"Check max pool. MSE: {mse}")

        z_reshape = self._reshape(z_pool)
        z_reshape_target = z_pool_target.view(-1, 12 * 12 * 20)
        mse = F.mse_loss(z_reshape, z_reshape_target)
        print(f"Check reshape. MSE: {mse}")

        z_fc1 = self._fc(z_reshape, self.fc1.weight, self.fc1.bias)
        z_fc1_target = self.fc1(z_reshape_target)
        mse = F.mse_loss(z_fc1, z_fc1_target)
        print(f"Check fc1. MSE: {mse}")

        z_relu = self._relu(z_fc1)
        z_relu_target = F.relu(z_fc1_target)
        mse = F.mse_loss(z_relu, z_relu_target)
        print(f"Check relu. MSE: {mse}")


        z_fc2 = self._fc(z_relu, self.fc2.weight, self.fc2.bias)
        z_fc2_target = self.fc2(z_relu_target)
        mse = F.mse_loss(z_fc2, z_fc2_target)
        print(f"Check fc2. MSE: {mse}")


        z_softmax = z_fc2.softmax(dim=1)
        z_softmax_target = z_fc2_target.softmax(dim=1)
        mse = F.mse_loss(z_softmax, z_softmax_target)
        print(f"Check softmax. MSE: {mse}")

        return z_softmax


    def _conv(self, A, W, bias):
        n_batch = A.shape[0]
        s_in = A.shape[2]

        c_out = W.shape[0]
        k = W.shape[2]

        s_out = s_in - k + 1

        z = np.zeros((n_batch, c_out, s_out, s_out))

        for n in range(0, n_batch):
            for c in range(0, c_out):
                for m in range(0, s_out):
                    for l in range(0, s_out):
                        z[n, c, m, l] = self._conv_helper(A[n], W[c], m, l) + bias[c]


        Z = torch.as_tensor(z, dtype=torch.float32)
        Z.requires_grad_()

        return Z

    def _max_pool_kernel2(self, x):
        n_batch = x.shape[0]
        b = x.shape[1]
        s_conv = x.shape[2]

        s_pool = s_conv // 2

        z = np.zeros((n_batch, b, s_pool, s_pool))

        for n in range(0, n_batch):
            for c in range(0, b):
                for m in range(0, s_pool):
                    for l in range(0, s_pool):
                        z[n, c, m, l] = torch.max(torch.tensor([x[n][c][2 * m][2 * l], x[n][c][2 * m][2 * l + 1],
                                            x[n][c][2 * m + 1][2 * l], x[n][c][2 * m + 1][2 * l + 1]]))

        Z = torch.as_tensor(z, dtype=torch.float32)
        Z.requires_grad_()
        return Z

    def _reshape(self, x):
        n_batch = x.shape[0]
        b = x.shape[1]
        s_pool = x.shape[2]

        n_reshape = b * s_pool * s_pool

        z = np.zeros((n_batch, n_reshape))

        for n in range(0, n_batch):
            for c in range(0, b):
                for m in range(0, s_pool):
                    for l in range(0, s_pool):
                        j = c * s_pool * s_pool + m * s_pool + l

                        z[n][j] = x[n][c][m][l]

        Z = torch.as_tensor(z, dtype=torch.float32)
        Z.requires_grad_()

        return Z

    def _fc(self, x, weight, bias):
        n_batch = x.shape[0]
        d = x.shape[1]

        p = weight.shape[0]

        z = np.zeros((n_batch, p))

        for n in range(0, n_batch):
            for j in range(0, p):
                sum = 0
                for i in range(0, d):
                    sum += weight[j][i] * x[n][i]

                z[n][j] = sum + bias[j]

        Z = torch.as_tensor(z, dtype=torch.float32)
        Z.requires_grad_()

        return Z

    def _relu(self, x):
        n_batch = x.shape[0]
        p = x.shape[1]

        z = np.zeros((n_batch, p))
        for n in range(0, n_batch):
            for i in range(0, p):
                z[n, i] = max(0, x[n, i])

        Z = torch.as_tensor(z, dtype=torch.float32)
        Z.requires_grad_()
        return Z

    def _conv_helper(self, x_n, w_c_out, m, l):
        c_in = x_n.shape[0]
        k = w_c_out.shape[1]

        value = 0
        for c in range(0, c_in):
            for i in range(0, k):
                for j in range(0, k):
                    value += x_n[c, m + i, l + j] * w_c_out[c, i, j]
        return value
