"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # assert out_grad.dtype == np.float32
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # assert out_grad.dtype == np.float32
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        result = a ** numpy.float32(self.scalar)
        return result

    def gradient(self, out_grad, node):
        result = out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1),
        # assert result[0].dtype == np.float32
        return result


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        result = a / b
        return result

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        result = (out_grad / rhs), negate((multiply(out_grad, lhs) / pow(rhs, 2)))
        # assert result[0].dtype == np.float32
        # assert result[1].dtype == np.float32
        return result


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        result = a / numpy.float32(self.scalar)
        return result

    def gradient(self, out_grad, node):
        result = divide_scalar(out_grad, self.scalar),
        # assert result[0].dtype == np.float32
        return result


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

# hbsun: please note Transpose can only support 2 axis now.
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            self.axes = tuple(range(len(a.shape) - 1, len(a.shape) - 3, -1))
        newAxis = numpy.arange(len(a.shape), dtype=int)
        for i in range(len(self.axes)):
            newAxis[self.axes[len(self.axes) - 1 - i]] = self.axes[i]
        result = a.permute(newAxis)
        return result

    def gradient(self, out_grad, node):
        result = transpose(out_grad, self.axes),
        # assert result[0].dtype == np.float32
        return result


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        result = reshape(out_grad, node.inputs[0].shape),
        # assert result[0].dtype == np.float32
        return result


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # len(a.shape) = 1 and len(self.shape) > 1:
        if len(a.shape) == 1 and len(self.shape) > 1:
            extended_shape = [1] * (len(self.shape) - 1) + list(a.shape)
            # print("BroadcastTo::extended_shape", extended_shape)
            result = a.reshape(tuple(extended_shape))
        else:
            result = a
        return result.broadcast_to(self.shape).compact() #hbsun: please note this compact

    def gradient(self, out_grad, node):
        # out_grad -> node.inputs[0]
        # shape(3,3,3) -> shape(1) by axis = (0,1,2)
        # shape(3,3) -> shape(3, 1) by axis = (1)
        # shape(3,3) -> shape(1, 3) by axis = (0)

        # For hw3:
        # shape(1,5) -> shape(5,) by axis = (0)
        # shape(3,5) -> shape(5,) by axis = (0)
        # shape(1,3,5) -> shape(5,) by axis = (0,1)

        # need to make sure the shape of out_grad is the same as the shape of node.inputs[0] by summation
        shape_out_grad = out_grad.shape
        lhs = Tensor(node.inputs[0])
        shape_lhs = list(lhs.shape)
        new_shape = []
        i = len(shape_out_grad) - 1
        j = len(shape_lhs) - 1
        while i >= 0:
            if j < 0 or shape_out_grad[i] != shape_lhs[j]:
                new_shape.append(i)
            i -= 1
            j -= 1
        new_shape.reverse()
        result = summation(out_grad, tuple(new_shape))
        result = reshape(result, shape_lhs)
        # assert result.dtype == np.float32
        return result


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return a.sum()

        if type(self.axes) is int:
            return a.sum(axis=self.axes)

        # reverse visit the axes
        for i in range(len(self.axes), 0, -1):
            a = a.sum(axis=self.axes[i - 1])
        return a

    def gradient(self, out_grad, node):
        # axes=1: (5, 4) -> (5, 1) -> (5, 4)
        # axes=0: (5, 4) -> (1, 4) -> (5, 4)
        lhs = Tensor(node.inputs[0])
        newShape = list(lhs.shape)
        if self.axes is not None:
            if type(self.axes) is int:
                self.axes = (self.axes,)
            for i in self.axes:
                newShape[i] = 1
            out_grad = reshape(out_grad, newShape)

        result = broadcast_to(out_grad, lhs.shape),
        # assert result[0].dtype == np.float32
        return result


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        result = a @ b
        return result

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)

        # below is for test_matmul_batched_backward
        lhs_len = len(lhs.shape)
        rhs_len = len(rhs.shape)
        if lhs_len > rhs_len:
            rhs_chang = range(lhs_len - rhs_len)
            rhs_grad = summation(rhs_grad, tuple(rhs_chang))
        elif lhs_len < rhs_len:
            lhs_chang = range(rhs_len - lhs_len)
            lhs_grad = summation(lhs_grad, tuple(lhs_chang))

        result = lhs_grad, rhs_grad
        # assert result[0].dtype == np.float32
        # assert result[1].dtype == np.float32
        return result


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        result = -a
        return result

    def gradient(self, out_grad, node):
        return negate(out_grad),


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        result = array_api.log(a)
        return result

    def gradient(self, out_grad, node):
        result = out_grad / node.inputs[0],
        # assert result[0].dtype == np.float32
        return result,


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        result = array_api.exp(a)
        return result

    def gradient(self, out_grad, node):
        lhs = Tensor(node.inputs[0])
        result = multiply(out_grad, exp(lhs)),
        # assert result[0].dtype == np.float32
        return result


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        result = array_api.maximum(a, numpy.float32(0))
        return result

    def gradient(self, out_grad:Tensor, node):
        out_grad_data = out_grad.detach()
        lhs_data = node.inputs[0].realize_cached_data()
        mask = Tensor(lhs_data > 0, device=out_grad.device)
        return multiply(out_grad_data, mask),


def relu(a):
    return ReLU()(a)


# axes - Tuple of axes to sum and take the maximum element over. If None, all axes are summed.
# Applies a numerically stable log-sum-exp function to the input by subtracting off the maximum elements.
# axes - Tuple of axes to sum and take the maximum element over. This uses the same conventions as needle.ops.Summation()
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.Z_max = None
        self.axes = axes

    def compute(self, Z):
        # hbsun: we need keepdims=True to make sure the shape of Z_max is the same as Z.
        # Otherwise, implicated broadcasting will be unexpected.
        Z_max = Z.max(self.axes, keepdims=True)
        Z_max = Z_max.broadcast_to(Z.shape)
        self.Z_max = Z_max
        result = (Z - Z_max).exp().sum(self.axes).log()
        result += Z.max(self.axes)
        return result

    def gradient(self, out_grad: Tensor, node):
        Z = node.inputs[0]
        new_shape = list(Z.shape)
        if self.axes is None:
            self.axes = tuple(range(len(new_shape)))
        if type(self.axes) is int:
            self.axes = (self.axes,)
        for i in self.axes:
            new_shape[i] = 1
        Z_max = Tensor(self.Z_max, device=Z.device)
        Z_reg = exp(Z - Z_max)
        denominator = summation(Z_reg, self.axes)
        grad = (Z_reg / broadcast_to(reshape(denominator, new_shape), Z.shape)).detach()
        out_grad = out_grad.reshape(new_shape).broadcast_to(Z.shape).detach()
        return out_grad * grad,


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        return a.tanh()

    def gradient(self, out_grad, node):
        lhs = Tensor(node.inputs[0])
        return multiply(out_grad, (- tanh(lhs) ** 2) + numpy.float32(1)),  # hbsun: 1 need to be added at the end.


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        len_args = len(args)
        if len_args == 0:
            raise ValueError("Need at least one array to stack.")

        shape = args[0].shape
        for i in range(1, len_args):
            if args[i].shape != shape:
                raise ValueError("All arrays must be of the same size.")

        new_shape = list(shape)
        new_shape.insert(0, len_args)
        result = array_api.empty(new_shape, device=args[0].device)

        for i in range(len_args):
            # use index to access result element
            index = [slice(0, len_shape, 1) for len_shape in shape]
            index.insert(0, slice(i, i + 1, 1))
            result[tuple(index)] = args[i]

        permute_axes = list(range(len(new_shape)))
        for i in range(0, self.axis + 1):
            permute_axes[i] = i + 1
        permute_axes[self.axis] = 0
        result = result.permute(permute_axes)
        return result

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis),


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        shape = A.shape
        if self.axis >= len(shape):
            raise ValueError("Axis must be less than the number of dimensions.")

        permute_axes = list(range(len(shape)))
        for i in range(self.axis + 1):
            permute_axes[i] = i - 1
        permute_axes[0] = self.axis
        result = A.permute(tuple(permute_axes))
        index = [slice(0, len_shape, 1) for len_shape in result.shape]

        ans = []
        for i in range(result.shape[0]):
            index[0] = slice(i, i+1, 1)
            tmp = result[tuple(index)].compact()
            newShape = list(tmp.shape)
            newShape.pop(0)
            tmp = tmp.reshape(newShape)
            ans.append(tmp)
        return ans

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis),


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        for i in self.axes:
            if i >= len(a.shape):
                return a

        dilation = self.dilation + 1
        dilated_shape = list(a.shape)
        for i in self.axes:
            dilated_shape[i] *= dilation
        result = array_api.empty(dilated_shape, device=a.device)
        result.fill(0)
        index = [slice(0, dilated_shape[j], 1) for j in range(len(dilated_shape))]
        for i in range(len(dilated_shape)):
            if i in self.axes:
                index[i] = slice(0, dilated_shape[i], dilation)
        result[tuple(index)] = a
        return result

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        dilation = int(self.dilation) + 1
        index = [slice(0, a.shape[j], 1) for j in range(len(a.shape))]
        for i in range(len(a.shape)):
            if i in self.axes:
                index[i] = slice(0, a.shape[i], dilation)
        return a[tuple(index)]

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # A: NDArray in NHWC format
        # B: NDArray in HWIO format
        # implement convolution using im2col
        fourPad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        A = A.pad(fourPad)

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * C_in
        if self.stride == 1:
            A = A.as_strided(shape=(N, H - K + 1, W - K + 1, K, K, C_in), strides=(Ns, Hs, Ws, Hs, Ws, Cs))
            A = A.compact().reshape((int((N * (H - K + 1) * (W - K + 1) * K * K * C_in) / inner_dim), inner_dim))
        else:
            A = A.as_strided(shape=(N, (H - K) // self.stride + 1, (W - K) // self.stride + 1, K, K, C_in),
                             strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs))
            A = A.compact().reshape((int((N * ((H - K) // self.stride + 1) * (
                        (W - K) // self.stride + 1) * K * K * C_in) / inner_dim), inner_dim))

        B = B.compact().reshape((inner_dim, C_out))
        out = A @ B
        out = out.reshape((N, (H - K)//self.stride + 1, (W - K)//self.stride+1, C_out))

        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        X, W = node.inputs
        K, _, _, C_out = W.shape

        # calculate X.grad first
        # flip W
        localW = flip(W, (0, 1)).detach()
        localW = localW.transpose((3, 2))

        localGrad = out_grad.detach()
        localGrad = dilate(localGrad, (1, 2), max(0, self.stride-1))

        X_grad = conv(localGrad, localW, padding=K - self.padding - 1)

        # calculate W.grad second
        # modify X and out_grad so that the shape of their convolution matches the shape of W
        localX = X.transpose((3, 0)).detach()
        localGrad = localGrad.transpose((1, 0))
        localGrad = localGrad.transpose((2, 1))
        W_grad = conv(localX, localGrad, padding=self.padding)
        W_grad = W_grad.transpose((1, 0))
        W_grad = W_grad.transpose((2, 1))

        return X_grad, W_grad


def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)



