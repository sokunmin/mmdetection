import torch
import torch.nn as nn

from ..builder import LOSSES
import random


@LOSSES.register_module
class BalanceLoss(nn.Module):
    """
        Class that implements automatically weighed loss from:
        https://arxiv.org/pdf/1705.07115.pdf
        https://github.com/AidasLiaudanskas/AutomaticLossWeightingPyTorch/blob/master/weighted_loss.py

        NOTE:
        Don't forget to give these params to the optimiser:
        optim.SGD(model.parameters() + criterion.parameters(), optim_args).

        `d`=`discrete`, `c`=`continuous`
        Parameters:
        -----------
        num_d: number of discrete losses
        num_c: number of continuous losses

    """

    def __init__(self,
                 num_discrt=1,
                 num_contin=1,
                 reduction='mean'):
        super(BalanceLoss, self).__init__()
        assert num_discrt > 0 or num_contin > 0
        self.reduction = reduction
        self.logvars_d = []
        self.logvars_c = []
        for i in range(num_discrt):
            param = self._init('d_', i)
            self.logvars_d.append(param)

        for i in range(num_contin):
            param = self._init('c_', i)
            self.logvars_c.append(param)

    def _init(self, name, index):
        init_value = random.random()
        param = nn.Parameter(torch.tensor(init_value))
        name = name + str(index)
        self.register_parameter(name, param)
        return param

    def forward(self,
                losses_d,
                losses_c,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(losses_d) == len(self.logvars_d)
        assert len(losses_c) == len(self.logvars_c)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        loss = losses_d[-1].new_tensor([0])
        # loss = torch.zeros(1).to(self.device)
        for i, loss_d in enumerate(losses_d):
            # > e^(-log_var)
            loss += 0.5 * torch.exp(-self.logvars_d[i]) * loss_d + 0.5 * self.logvars_d[i]
            # > e^(-2 * og_var)
            # loss += 0.5 * torch.exp(-self.logvars_d[i]) * loss_d + 0.25 * self.logvars_d[i]
            # loss += 0.5 * torch.exp(-2 * self.logvars_d[i]) * loss_d + 0.5 * self.logvars_d[i]

        for i, loss_c in enumerate(losses_c):
            loss += torch.exp(-self.logvars_c[i]) * loss_c + 0.5 * self.logvars_c[i]

        return dict(loss_total=loss)
