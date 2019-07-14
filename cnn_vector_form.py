import torch
import torch.nn as nn

import torch.nn.functional as F


class CNNVectorForm(nn.Module):

    def __init__(self, check_with_pytorch):
        super(CNNVectorForm, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.fc1 = nn.Linear(12 * 12 * 20, 500)
        self.fc2 = nn.Linear(500, 10)

        self.check_with_pytorch = check_with_pytorch


    def forward(self, x):
        if self.check_with_pytorch:
            return self._forward_with_cmp(x)

        return self._forward(x)


    def _forward(self, x):
        z_conv = self._conv2d_vector_bonus(x, self.conv1.weight, self.conv1.bias)
        z_pool = self._max_pool(z_conv, 2, 2)

        z_reshape = z_pool.view(z_pool.shape[0], z_pool.shape[1] * z_pool.shape[2] * z_pool.shape[3])

        z_fc1 = self._fc(z_reshape, self.fc1.weight, self.fc1.bias)

        z_relu = z_fc1.relu()

        z_fc2 = self._fc(z_relu, self.fc2.weight, self.fc2.bias)

        z_softmax = z_fc2.softmax(dim=1)

        return z_softmax


    def _forward_with_cmp(self, x):
        z_conv = self._conv2d_vector_bonus(x, self.conv1.weight, self.conv1.bias)
        z_conv_target =  self.conv1(x)
        mse = F.mse_loss(z_conv, z_conv_target)
        print(f"Check conv. MSE: {mse}")

        z_pool = self._max_pool(z_conv, 2, 2)
        z_pool_target = F.max_pool2d(z_conv_target, 2, 2)
        mse = F.mse_loss(z_pool, z_pool_target)
        print(f"Check max pool. MSE: {mse}")

        z_reshape = z_pool.view(z_pool.shape[0], z_pool.shape[1] * z_pool.shape[2] * z_pool.shape[3])
        z_reshape_target = z_pool_target.view(-1, 12 * 12 * 20)
        mse = F.mse_loss(z_reshape, z_reshape_target)
        print(f"Check reshape. MSE: {mse}")

        z_fc1 = self._fc(z_reshape, self.fc1.weight, self.fc1.bias)
        z_fc1_target = self.fc1(z_reshape_target)
        mse = F.mse_loss(z_fc1, z_fc1_target)
        print(f"Check fc1. MSE: {mse}")

        z_relu = z_fc1.relu()
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



    def _im2col_KK(self, x, k, s):
        c_in = x.shape[0]
        s_in = x.shape[1]

        s_out = (s_in - k) // s + 1

        h_col = k * k
        w_col = c_in * s_out * s_out

        X_col = torch.zeros(h_col, w_col, requires_grad=True)
        i = 0
        for c in range(0, c_in):
            for m in range(0, s_in - k + 1, s):
                for l in range(0, s_in - k + 1, s):
                    x_col_i = self._mat2row(x[c, m:m + k, l:l + k])
                    X_col[:, i] = x_col_i
                    i += 1

        return X_col

    def _weight2row_kk(self, wc_conv):
        c_out = wc_conv.shape[0]
        k = wc_conv.shape[1]

        wc_row_conv = torch.zeros(c_out, k * k, requires_grad=True)
        for j in range(0, c_out):
            wc_row_j = self._mat2row(wc_conv[j, :, :])
            wc_row_conv[j, :] = wc_row_j

        return wc_row_conv

    def _max_pool(self, A, k, s):
        n_batch = A.shape[0]
        c_in = A.shape[1]
        s_in = A.shape[2]

        s_out = (s_in - k) // s + 1

        Z = torch.zeros(n_batch, c_in * s_out * s_out, requires_grad=True)
        for n in range(0, n_batch):
            A_col_n = self._im2col_KK(A[n], k, s)

            Z[n] = A_col_n.t().max(1).values

        Z_pool = Z.view(n_batch, c_in, s_out, s_out)

        return Z_pool

    # bonus fun

    def _conv2d_vector_bonus(self, A, W, bias, s=1):
        n_btach = A.shape[0]
        c_in = A.shape[1]
        s_in = A.shape[2]

        c_out = W.shape[0]
        k = W.shape[2]

        s_out = s_out = (s_in - k) // s + 1

        B = torch.zeros(c_out, s_out, s_out, requires_grad=True)
        for i in range(0, c_out):
            B[i] = bias[i]

        Z = torch.zeros(n_btach, c_out, s_out, s_out, requires_grad=True)

        for n in range(0, n_btach):
            W_row_n = self._weight2row_bonus(W)
            A_col_n = self._im2col_bonus(A[n], k, s)

            O_mat_n = torch.mm(W_row_n, A_col_n)

            O_n = O_mat_n.view((c_out, s_out, s_out))

            Z_n = O_n + B

            Z[n] = Z_n

        return Z

    def _weight2row_bonus(self, W):
        c_out = W.shape[0]  # num of filters
        c_in = W.shape[1]
        k = W.shape[2]

        W_row = torch.zeros(c_out, c_in * k * k, requires_grad=True)
        for j in range(0, c_out):
            W_row[j, :] = self._tensor2row(W[j, :, :, :])

        return W_row

    def _im2col_bonus(self, X, k, s):
        c_in = X.shape[0]
        s_in = X.shape[1]

        s_out = (s_in - k) // s + 1

        h_col = c_in * k * k
        w_col = s_out * s_out

        X_col = torch.zeros(h_col, w_col, requires_grad=True)

        i = 0
        for m in range(0, s_in - k + 1, s):
            for l in range(0, s_in - k + 1, s):
                X_col[:, i] = self._tensor2row(X[:, m:m + k, l:l + k])
                i += 1

        return X_col

    def _tensor2row(self, R):
        r = self._tensor2col(R).t()
        return r

    def _tensor2col(self, C):
        N = C.shape[0]
        M = C.shape[1]
        L = C.shape[2]

        c = torch.zeros(N * M * L, 1, requires_grad=True)

        for j in range(0, N):
            for i in range(0, M):
                for k in range(0, L):
                    t = j * M * L + i * L + k
                    c[t] = C[j, i, k]

        return c

    def _mat2row(self, C):

        N = C.shape[0]
        M = C.shape[1]

        # TODO: think about it
        # c = C.view(N * M, 1)

        c = torch.zeros(M * N, requires_grad=True)

        for j in range(0, N):
            for i in range(0, M):
                t = j * M + i

                c[t] = C[j][i]

        return c

    def _fc(self, A, W, bias):
        n_batch = A.shape[0]

        ones = torch.ones(n_batch, requires_grad=True).view(n_batch).tolist()

        A_temp = A.t().tolist()
        A_temp.append(ones)
        A_new = torch.tensor(A_temp, requires_grad=True).t()

        W_temp = W.t().tolist()
        W_temp.append(bias.tolist())
        W_new = torch.tensor(W_temp, requires_grad=True)

        Z = torch.mm(A_new, W_new)

        return Z
