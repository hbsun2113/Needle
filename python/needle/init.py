import math
import needle as ndl


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = ndl.default_device() if device is None else device
    array = device.rand(*shape, dtype=dtype) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = ndl.default_device() if device is None else device
    array = device.randn(*shape, dtype=dtype) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = ndl.default_device() if device is None else device
    array = device.full(shape, c, dtype=dtype)
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = ndl.default_device() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = ndl.default_device() if device is None else device
    return ndl.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def xavier_uniform_conv(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    k, _, c_in, c_out = shape
    bound = 1.0 / math.sqrt(c_in * k * k)
    newShape = (c_out, )
    return rand(*newShape, low=-bound, high=bound, **kwargs)


def kaiming_uniform_conv(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # common implementation of kaiming uniform
    # gain = math.sqrt(2.0)
    # bound = gain * math.sqrt(3.0 / fan_in)
    # return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)

    # special implementation of kaiming uniform for conv2d
    k = fan_in // shape[2]
    gain = math.sqrt(2.0 / k)
    bound = gain * math.sqrt(3.0 / (fan_in//k))
    return rand(*shape, low=-bound, high=bound, **kwargs)


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(*(fan_in, fan_out), low=-bound, high=bound, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(*(fan_in, fan_out), std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    gain = math.sqrt(2.0)
    bound = gain * math.sqrt(3.0 / fan_in)
    return rand(*(fan_in, fan_out), low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_in)
    return randn(*(fan_in, fan_out), std=std, **kwargs)


def rnn_uniform(fan_in, fan_out, k, **kwargs):
    bound = math.sqrt(k)
    return rand(*(fan_in, fan_out), low=-bound, high=bound, **kwargs)