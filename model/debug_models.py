import torch
from torch import nn


class TwoCircles(nn.Module):
    def __init__(self,
                 centers,
                 slide=0.1,
                 param_dim=1,
                 is_classification_head=False,
                 is_sdf=False,
                 overlap=False):
        super().__init__()
        self.centers = centers
        assert self.centers.shape == (2, 2)
        # 1, 1, 2, 2
        self.centers = self.centers.unsqueeze(0).unsqueeze(0)
        self.primitive_num = 2
        self.slide = slide
        self.is_classification_head = is_classification_head
        self.is_sdf = is_sdf
        self.param_dim = param_dim
        self.overlap = overlap

    def forward(self, inputs, points=None, **kwargs):
        self.centers = self.centers.to(points.device)
        if self.is_classification_head:
            occs = []
            for idx in range(self.primitive_num):
                # b, p, 1, 2
                radius = ((points -
                           self.centers[:, :, idx, :])**2).sum(-1).sqrt()
                occ = 0.2 - radius
                occs.append(occ)
            occs_t = torch.stack(occs, axis=-1)
            occs_sum = torch.relu(occs_t).sum(axis=-1)
            back = torch.where(occs_sum > 0,
                               torch.ones_like(occs_sum) * -1,
                               torch.ones_like(occs_sum) * 100)
            occs.insert(0, back)
            occ = torch.stack(occs, axis=-1)

        else:
            # b, p, 1, 2
            points = points.unsqueeze(-2)
            radius = ((points - self.centers)**2).sum(-1).sqrt()
            if self.is_sdf:
                occ = -(0.2 - radius)
            else:
                occ = 0.2 - radius
        slide = self.centers[0, 0, 1, 1] - self.centers[0, 0, 0, 1]
        slide = slide.unsqueeze(-1).expand(points.size(0), -1).to(
            points.device) * 0 + 0.3
        slide = slide.to(points.device)
        if self.param_dim == 2:
            slide = torch.cat([torch.zeros_like(slide), slide], axis=-1)
        if self.overlap:
            mask0 = (occ.argmax(-1) == 0) & (occ[:, :, 0] >= 0.)
            mask1 = (occ.argmax(-1) == 1) & (occ[:, :, 1] >= 0.)
            occ[:, :, 1] = (mask1.float() - 0.5) * 2
            occ[:, :, 0] = (mask0.float() - 0.5) * 2
            occ[:, :, 1] += mask0 * 1.1
            occ[:, :, 0] = -1.1
            #occ[:, :, 1] += occ[:, :, 0] * 0.9

        return {'occupancy': occ, 'param': slide}
