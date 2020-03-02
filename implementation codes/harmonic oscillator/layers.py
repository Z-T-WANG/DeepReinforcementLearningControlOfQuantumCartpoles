import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import math

class FactorizedNoisy(nn.Module):
    """
    Modified from https://jsapachehtml.hatenablog.com/entry/2018/10/13/173303
    """
    def __init__(self, in_features, out_features):
        super(FactorizedNoisy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.u_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self.weight_norm = Parameter(torch.tensor(1.))
        self.reset_parameters()
        self.noisy = True

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u_w.size(1))
        self.u_w.data.uniform_(-stdv, stdv)
        self.u_b.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)

    def forward(self, x):
        if not hasattr(self, 'randbuffer_in') or self.randbuffer_in.device!=self.u_w.device:
            self.randbuffer_in=self._f(torch.randn(2048, 1, self.in_features, device=self.u_w.device))
            self.randbuffer_out=self._f(torch.randn(2048, self.out_features, 1, device=self.u_w.device))
            self.randbuffer_pointer=0
        assert self.randbuffer_pointer != 2048
        i = self.randbuffer_pointer
        if x.size(0) % 128 != 0:
            n = x.size(0)
            x = x.view(n,-1,x.size(1))
            if hasattr(self, 'noisy'):
                if not self.noisy:
                    return F.linear(x, self.u_w, self.u_b)
            if i+n > 2048:
                self.randbuffer_in, self.randbuffer_out = self._f(torch.randn(2048, 1, self.in_features, device=self.u_w.device)), self._f(torch.randn(2048, self.out_features, 1, device=self.u_w.device))
                i=self.randbuffer_pointer = 0
            rand_in, rand_out = self.randbuffer_in[i:i+n,:,:], self.randbuffer_out[i:i+n,:,:]
            self.randbuffer_pointer += n
            epsilon_w = torch.bmm(rand_out, rand_in)
            epsilon_b = rand_out.squeeze(dim=2)

            if hasattr(self, 'weight_norm'):
                w = self.u_w/self.u_w.norm()*self.weight_norm + self.sigma_w * epsilon_w
            else:
                w = self.u_w + self.sigma_w * epsilon_w
            b = self.u_b + self.sigma_b * epsilon_b
            b.unsqueeze_(dim=1)
            output = torch.baddbmm(b, x, w.transpose(1,2))
            output = output.view(-1,output.size(2))
        else:
            n=32
            assert i+n <= 2048
            # separate input into n chunks for n random samplings. (x.size(0)%n==0)
            x = x.view(n,-1,x.size(1))
            rand_in, rand_out = self.randbuffer_in[i:i+n,:,:], self.randbuffer_out[i:i+n,:,:]
            self.randbuffer_pointer += n
            epsilon_w = torch.bmm(rand_out, rand_in)
            epsilon_b = rand_out.squeeze(dim=2)

            w = self.u_w/self.u_w.norm()*self.weight_norm + self.sigma_w * epsilon_w 
            # the first dimension of epsilon_w is the random sampling batch,
            b = self.u_b + self.sigma_b * epsilon_b # which equals the second dimension of x
            b.unsqueeze_(dim=1)
            output = torch.baddbmm(b, x, w.transpose(1,2))
            output = output.view(-1,output.size(2))

        if self.randbuffer_pointer + 32 > 2048: 
            self.randbuffer_in, self.randbuffer_out = self._f(torch.randn(2048, 1, self.in_features, device=self.u_w.device)), self._f(torch.randn(2048, self.out_features, 1, device=self.u_w.device))
            self.randbuffer_pointer = 0
        return output

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

from torch.nn.parameter import Parameter
class Conv1d_weight_normalize(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv1d_weight_normalize, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight_norm = Parameter(torch.tensor(1.))
    def forward(self, input):
        return F.conv1d(input, self.weight/self.weight.norm()*self.weight_norm, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Linear_weight_normalize(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_weight_normalize, self).__init__(in_features, out_features, bias)
        self.weight_norm = Parameter(torch.tensor(1.))
    def forward(self, input):
        return F.linear(input, self.weight/self.weight.norm()*self.weight_norm, self.bias)

class BatchRenorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-8, momentum=0.9998, affine=True,
                 track_running_stats=True):
        super(BatchRenorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.track_running_stats: self.running_mean[:]=0.;self.running_var[:]=0.
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        # the only way to implement batch renormalization in pytorch is to handwrite it
        correction = False
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            input_=input.detach()
            mean = input_.mean(dim=2).mean(dim=0)
            var = ((input_-mean.unsqueeze(dim=1))**2).sum(dim=2).sum(dim=0).div_(input_.size(0)*input_.size(2)-1).sqrt_()
            
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
                correction = 1. - (1.-self.momentum)**self.num_batches_tracked.item()
            torch.add(exponential_average_factor*self.running_mean, (1-exponential_average_factor), mean, out = self.running_mean)
            torch.add(exponential_average_factor*self.running_var, (1-exponential_average_factor), var, out = self.running_var)
            if correction!=1.:
                output = (input - self.running_mean.unsqueeze(dim=1)/correction) / (self.running_var.unsqueeze(dim=1)/correction+self.eps)
            else: output = (input - self.running_mean.unsqueeze(dim=1))/ (self.running_var.unsqueeze(dim=1)+self.eps)
            output=torch.addcmul(self.bias.unsqueeze(dim=1), 1, self.weight.unsqueeze(dim=1), output);
            return output
        else:
            correction = 1. - self.momentum**self.num_batches_tracked.item()
            # when self.num_batches_tracked.item()==0, it is singular and needs special treatment:
            if correction == 0.: self.running_mean[:]=0.; self.running_var[:]=1.; correction=1.

            if correction != 1.:
                return F.batch_norm(
                    input, self.running_mean/correction, (self.running_var)/correction, self.weight, self.bias,
                    self.training or not self.track_running_stats,
                    self.momentum, self.eps)
            else: 
                return F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training or not self.track_running_stats,
                    self.momentum, self.eps)

