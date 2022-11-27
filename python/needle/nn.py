"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose())

    def forward(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.weight.shape[0]:
            print("Shape mismatch, expected", self.weight.shape[0], "got", X.shape[1])
            X = ops.reshape(X, (-1, self.weight.shape[0]))
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out = ops.add(out, self.bias.broadcast_to(out.shape))
        return out



class Flatten(Module):
    def forward(self, X):
        # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html, newshape can be -1.
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            # print("m:", m)
            x = m(x)
            # if x.dtype != "float32":
            #     print("failed", m.__class__.__name__)
            # else:
            #     print("succee", m.__class__.__name__)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # hbsun: https://forum.dlsyscourse.org/t/when-will-the-tensors-value-be-calculated/2172/4
        # We have to divide scalar at the beginning. Because logits is a tensor with shape [M,N]. Then we can get a Tensor of float32.
        # Otherwise, if we divide the Tensor only contains one element by another scalar, we will get a Tensor of float64.
        # The final solution: https://forum.dlsyscourse.org/t/q3-tensor-dtype-mismatch/2297/3
        log_sum_exp = ops.logsumexp(logits, 1)
        y_one_hot = init.one_hot(logits.shape[1], y)
        Zy = ops.summation(ops.multiply(logits, y_one_hot), 1)
        loss = ops.summation(ops.negate(Zy) + log_sum_exp, 0)
        loss = loss / Tensor(logits.shape[0], dtype=loss.dtype, device=loss.device).data
        return loss




class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            Ex = ops.summation(x, 0) / Tensor(x.shape[0], dtype=x.dtype, device=x.device).data
            self.running_mean = ops.mul_scalar(self.running_mean, 1 - self.momentum).data + ops.mul_scalar(Ex, self.momentum).data
            Ex = ops.reshape(Ex, (1, x.shape[1]))
            Ex = ops.broadcast_to(Ex, x.shape)

            Vx = ops.summation(ops.power_scalar(x + ops.negate(Ex), 2), 0) / Tensor(x.shape[0], dtype=x.dtype, device=x.device).data
            self.running_var = ops.mul_scalar(self.running_var, 1 - self.momentum).data + ops.mul_scalar(Vx, self.momentum).data
            Vx = ops.reshape(Vx, (1, x.shape[1]))
            Vx = ops.broadcast_to(Vx, x.shape)

            denominator = ops.power_scalar(Vx + self.eps, 0.5)
            x_hat = ops.divide(x + ops.negate(Ex), denominator)

            # hbsun: we need call broadcast_to explicitly. Because if we use numpy.broadcast_to implicitly,
            # Needle backward will not be aware this call, and will not produce the internal node.
            weight = ops.broadcast_to(self.weight, x_hat.shape)
            bias = ops.broadcast_to(self.bias, x_hat.shape)
            result = ops.add(ops.multiply(x_hat, weight), bias)
            return result
        else:
            self.running_mean = ops.reshape(self.running_mean, (-1, x.shape[1]))

            # hbsun: here we cannot reassign self.running_mean, because x.shape maybe change.
            # For example, the batch size is 150. But the last batch is only 100.
            running_mean = ops.broadcast_to(self.running_mean, x.shape).data
            self.running_var = ops.reshape(self.running_var, (-1, x.shape[1]))
            running_var = ops.broadcast_to(self.running_var, x.shape).data
            denominator = ops.power_scalar(running_var + self.eps, 0.5)
            x_hat = ops.divide(x + ops.negate(running_mean), denominator)

            # Different from training mode, we can use numpy.broadcast_to implicitly because there is no backward.
            result = ops.add(ops.multiply(x_hat, self.weight), self.bias)
            assert x.dtype == result.dtype
            return result


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        Ex = ops.summation(x, 1) / Tensor(x.shape[1], dtype=x.dtype, device=x.device).data
        Ex = ops.reshape(Ex, (x.shape[0], 1))
        Ex = ops.broadcast_to(Ex, x.shape)
        Vx = ops.summation(ops.power_scalar(x + ops.negate(Ex), 2), 1) / Tensor(x.shape[1], dtype=x.dtype, device=x.device).data
        Vx = ops.reshape(Vx, (x.shape[0], 1))
        Vx = ops.broadcast_to(Vx, x.shape)
        denominator = ops.power_scalar(Vx + self.eps, 0.5)
        x_hat = ops.divide(x + ops.negate(Ex), denominator)

        # hbsun: same as BatchNorm1d
        weight = ops.broadcast_to(self.weight, x_hat.shape)
        bias = ops.broadcast_to(self.bias, x_hat.shape)

        res = ops.add(ops.multiply(x_hat, weight), bias)
        return res


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p = 1 - self.p)
            result = ops.multiply(x, mask)
            result = ops.divide_scalar(result, 1 - self.p)
        else:
            result = x
        return result

# hbsun: the below link is a very very simple/short introduction to the concept of "Residual" in deep learning.
# hbsun: https://zhuanlan.zhihu.com/p/22447440
# hbsun: https://www.cnblogs.com/wuliyttaotao/p/9560205.html
class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class Add(Module):
    def __init__(self, fn1: Module, fn2: Module):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2

    def forward(self, x: Tensor) -> Tensor:
        return self.fn1(x) + self.fn2(x)

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = Parameter(init.kaiming_uniform(fan_in=in_channels*kernel_size*kernel_size,
                                                     fan_out=out_channels*kernel_size*kernel_size,
                                                     shape=(kernel_size, kernel_size, in_channels, out_channels),
                                                     device=device,
                                                     dtype=dtype))

        if bias:
            notUsed = 1
            self.bias = Parameter(init.xavier_uniform(fan_in=notUsed,
                                                      fan_out=notUsed,
                                                      shape=(kernel_size, kernel_size, in_channels, out_channels),
                                                      gain=notUsed,
                                                      device=device,
                                                      dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # N, C, H, W = x.shape
        # print("N:", N, "C:", C, "H:", H, "W:", W)

        # from nchw to nhwc
        x = x.transpose((1, 2)).transpose((2, 3))

        # add padding to ensure input and output dimensions are the same
        padSize = self.kernel_size - 1
        padSize = padSize // 2
        result = ops.conv(a=x, b=self.weight, stride=self.stride, padding=padSize)

        # self.bias is a 1D tensor with shape(out_channels,), so we need to broadcast it to 4D
        if self.bias is not None:
            bias = ops.broadcast_to(self.bias, result.shape)
            result = ops.add(result, bias)

        return result.transpose((2, 3)).transpose((1, 2))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
