from torch import nn
from torch.nn import functional as F
import torch


class generator(nn.Module):
    def __init__(self,
                 z_dim,
                 point_dim,
                 gf_dim,
                 out_dim=1,
                 last_act_type='imnet_original'):
        super(generator, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.out_dim = out_dim
        self.last_act_type = last_act_type
        assert self.last_act_type in ['imnet_original', 'none']
        self.linear_1 = nn.Linear(self.z_dim + self.point_dim,
                                  self.gf_dim * 8,
                                  bias=True)
        self.linear_2 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim * 8, self.gf_dim * 4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim * 4, self.gf_dim * 2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim * 2, self.gf_dim * 1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim * 1, self.out_dim, bias=True)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias, 0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias, 0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias, 0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, points, z, is_training=False, return_act=False):
        zs = z.view(-1, 1, self.z_dim).repeat(1, points.size()[1], 1)
        pointz = torch.cat([points, zs], 2)

        l1 = self.linear_1(pointz)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)

        #l7 = torch.clamp(l7, min=0, max=1)
        if self.last_act_type == 'imnet_original':
            l7 = torch.max(torch.min(l7, l7 * 0.01 + 0.99), l7 * 0.01)

        if return_act:
            ret = {
                'l1': l1,
                'l2': l2,
                'l3': l3,
                'l4': l4,
                'l5': l5,
                'l6': l6,
                'occ': l7
            }
            return ret

        return l7
