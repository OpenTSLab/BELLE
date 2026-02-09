import torch
import torch.nn.functional as F
import numpy as np

def MSE(y, y_, reduce=True):
    mse = torch.mean((y - y_)**2, dim=list(range(1, len(y.shape))))
    return torch.mean(mse) if reduce else mse

def RMSE(y, y_):
    rmse = torch.sqrt(torch.mean((y - y_)**2))
    return rmse

def Gaussian_NLL(y, mu, sigma, reduce=True):
    logprob = -torch.log(sigma) - 0.5*torch.log(torch.tensor(2*np.pi)) - 0.5*((y - mu) / sigma)**2
    loss = torch.mean(-logprob, dim=list(range(1, len(y.shape))))
    return torch.mean(loss) if reduce else loss

def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    log_likelihood = 0.5 * (
        -torch.exp(-logvar) * (mu - y)**2 - torch.log(2 * torch.tensor(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = torch.mean(-log_likelihood, dim=list(range(1, len(y.shape))))
    return torch.mean(loss) if reduce else loss

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)
    
    nll = 0.5 * torch.log(torch.tensor(np.pi) / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1)**2) \
        + 0.5 * v2 / v1 \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    error = torch.abs(y - gamma)
    
    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + alpha
        # evi = 1 / (beta * (1 + v) / (v * (alpha - 1)))
        reg = error * evi

    return torch.mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = torch.chunk(evidential_output, 4, dim=-1)
    v = F.softplus(v)
    alpha = F.softplus(alpha) + 1
    beta = F.softplus(beta)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta, reduce=False)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, reduce=False)
    return loss_nll + coeff * loss_reg, gamma, v, alpha, beta
