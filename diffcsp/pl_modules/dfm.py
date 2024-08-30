# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class DFM(object):

    def __init__(self, n_types, base='mask', max_t=1000.0):

        # Set parameters
        self.max_t = max_t
        self.mask_token = n_types + 1

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
            t_scaled = t / self.max_t
            mask = torch.rand_like(atoms) < (1 - t_scaled[:, None])
            atoms[mask] = self.mask_token

        # Return atoms
        return atoms

class RateMatrix(object):

    def __init__(self, S, noise=0, switch='mask'):

        # Set switch for masking or uniform rate matrix and noise
        self.S = S
        self.noise = noise
        self.switch = switch.lower()
        if self.switch not in ('mask', 'uniform'):
            raise NotImplementedError

    def __call__(self, x_t, x_1, t):

        # Compute probabilities and derivatives based on prior
        if self.switch == 'mask':
            x_1_hot = F.one_hot(x_1, num_classes=self.S)
            M_hot = F.one_hot(torch.tensor([self.S-1]), num_classes=self.S)[None, :, :]
            dpt = x_1_hot - M_hot
            dpt_xt = dpt.gather(-1, x_t[:,:,None]).squeeze(-1)
            pt = (t * x_1_hot) + (1-t) * M_hot

        elif self.switch == 'uniform':
            x_1_hot = F.one_hot(x_1, num_classes=self.S)
            dpt = x_1 - (1 / self.S)
            dpt_xt = dpt.gather(-1, x_t[:,:,None]).squeeze(-1)
            pt = (t * x_1_hot) + (1-t) * (1 / self.S)

        # Compute and return rate
        pt_xt = pt.gather(-1, x_t[:,:,None]).squeeze(-1)
        Z = torch.count_nonzero(pt, dim=-1)
        R = F.relu(dpt - dpt_xt[:,:,None]) / ((Z * pt_xt)[:,:,None])
        R[(pt_xt == 0.0)[:,:,None].repeat(1, 1, self.S)] = 0.0
        R[pt == 0.0] = 0.0
        return R