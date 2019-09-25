import math
import torch
from torch.optim import Optimizer


class LARAdamW(Optimizer):
    r"""Implements Lookahead optimizer wrapped around RAdamW optimizer

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    The RAdam variant was proposed in `On The Variance of The Adaptive Learning Rate and Beyond`_.
    The Lookahead optimizer was proposed in `Lookahead Optimizer: k steps forward, 1 step back`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        k (int, optional): number of steps in synchronization period (default: 5)
        alpha (float, optional): slow weights step size should be between 0 and 1
            (default: 0.5)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On The Variance of The Adaptive Learning Rate and Beyond:
        https://arxiv.org/abs/1908.03265
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _Lookahead Optimizer\: k steps forward, 1 step back:
        https://arxiv.org/abs/1907.08610
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, k = 5, alpha = 0.5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= alpha < 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, k=k,
                        alpha=alpha)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['slow_p'] = p.clone().detach()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                sma_max = 2 / (1 - beta2) - 1
                sma = sma_max - 2 * t * beta2 ** t / (1 - beta2 ** t)
                if sma > 4:
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        # Specifying out ensures that state['max_exp_avg_sq'] is changed
                        # If one uses max_exp_avg_sq = torch.max(max_exp_avg_sq, exp_avg_sq)
                        # new Tensor is created and assigned to max_exp_avg_sq and
                        # state['max_exp_avg_sq'] is not changed
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    rt = math.sqrt((sma-4)/(sma_max-4)*(sma-2)/(sma_max-2)*sma_max/sma)
                    step_size = group['lr'] * rt / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr']/bias_correction1
                    p.data.add_(-step_size, exp_avg)
                k, alpha = group['k'], group['alpha']
                slow_p = state['slow_p']
                if t % k == 0:
                    slow_p.add_(alpha, p - slow_p)
                    p.data.copy_(slow_p)
        return loss
