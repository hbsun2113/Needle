"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    # hbsun: for momentum with weight decay:
    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    # https://alband.github.io/doc_view/_modules/torch/optim/sgd.html#SGD
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.detach()

            if self.weight_decay != 0.0:
                grad += self.weight_decay * p.data

            if self.momentum != 0.0:
                if p not in self.u:
                    self.u[p] = (1 - self.momentum) * grad.data
                else:
                    self.u[p] = self.momentum * self.u[p].data + (1 - self.momentum) * grad.data
                grad = self.u[p].data

            p.data -= self.lr * grad.data

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        bias_correction=True,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.bias_correction = bias_correction
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.detach()

            if self.weight_decay != 0.0:
                grad += self.weight_decay * p.data

            if p not in self.m:
                self.m[p] = grad.data * (1 - self.beta1)
            else:
                self.m[p] = self.beta1 * self.m[p].data + (1 - self.beta1) * grad.data

            if p not in self.v:
                self.v[p] = (grad.data ** 2) * (1 - self.beta2)
            else:
                self.v[p] = self.beta2 * self.v[p].data + (1 - self.beta2) * (grad.data ** 2)

            if self.bias_correction:
                m_hat = self.m[p] / (1 - self.beta1 ** self.t)
                v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            else:
                m_hat = self.m[p]
                v_hat = self.v[p]

            p.data -= self.lr * m_hat.data / (ndl.ops.power_scalar(v_hat.data, 0.5) + self.eps)


