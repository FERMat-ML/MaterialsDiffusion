# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class DFM(object):

    def __init__(self, n_types, base='mask'):

        # Set parameters
        self.mask_token = 0

        # Check for implemented absorbing state
        if base not in ['mask', 'uniform']:
            raise NotImplementedError
        else:
            self.base = base

    def mask(self, atoms, t):
        '''
        Mask atoms
        '''
        
        if self.base == 'mask':

            # Mask atoms based on t
            mask = torch.rand_like(atoms) < (1 - t[:, None])
            atoms[mask] = self.mask_token

        # Return atoms
        return atoms

class RateMatrix(object):

    def __init__(self, S, device, noise=0, switch='mask'):

        # Set switch for masking or uniform rate matrix and noise
        self.S = S
        self.noise = noise
        self.switch = switch.lower()
        self.device = device
        if self.switch not in ('mask', 'uniform'):
            raise NotImplementedError

    def __call__(self, x_t, x_1, t):

        # Compute probabilities and derivatives based on prior
        print(x_1)
        x_t = x_t.to(self.device)
        x_1 = x_1.to(self.device)
        if self.switch == 'mask':
            x_1_hot = F.one_hot(x_1, num_classes=self.S).to(self.device)
            M_hot = F.one_hot(torch.tensor([0]), num_classes=self.S)[None, :, :].to(self.device)
            dpt = (x_1_hot - M_hot).to(self.device)
            dpt_xt = dpt.gather(-1, x_t[:,:,None]).squeeze(-1)
            pt = (t * x_1_hot) + (1-t) * M_hot

        elif self.switch == 'uniform':
            x_1_hot = F.one_hot(x_1, num_classes=self.S)
            dpt = x_1 - (1 / self.S)
            dpt_xt = dpt.gather(-1, x_t[:,:,None]).squeeze(-1)
            pt = (t * x_1_hot) + (1-t) * (1 / self.S)

        # Compute and return rate
        pt = pt.to(self.device)
        pt_xt = pt.gather(-1, x_t[:,:,None]).squeeze(-1)
        pt_xt = pt_xt.to(self.device)
        Z = torch.count_nonzero(pt, dim=-1).to(self.device)
        R = F.relu(dpt - dpt_xt[:,:,None]) / ((Z * pt_xt)[:,:,None])
        R[(pt_xt == 0.0)[:,:,None].repeat(1, 1, self.S)] = 0.0
        R[pt == 0.0] = 0.0
        return R
