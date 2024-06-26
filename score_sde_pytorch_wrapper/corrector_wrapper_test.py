# Import libraries
import sys
import os
import unittest
import numpy as np
import torch

# Alter path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predictor_wrapper import PredictorWrapper

class Batch:
    def __init__(self, atom_types, num_atoms):
        self.atom_types = atom_types
        self.num_atoms = num_atoms
        self.batch = None

class TestPredictor(unittest.TestCase):
    def setUp(self):

        # Decoder/vars for testing purposes
        self.decoder = lambda t, a, x, l, natoms, batch: (1, 1)
        self.lr = 0.001
        self.sigma_norm = torch.tensor([1])
        self.x_t = torch.tensor([0.0])
        self.l_t = torch.tensor([0.0])

        # Initialize predictor
        self.corrector_wrapper = CorrectorWrapper(
            predictor_name_x="ancestral_sampling", predictor_name_l="ancestral_sampling", 
            sde_name_x="vesde", sde_name_l="vpsde", decoder=self.decoder, step_lr_x=self.lr, 
            step_lr_l=self.lr, sigma_min=0.005, sigma_max=0.5, 
            N=1000, beta_min=0.1, beta_max=20, number_predictor_steps=1
        )
        self.sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(0.005), np.log(0.5), 1000)))
    
    def test_pred(self):

        # Save RNG state
        state = torch.get_rng_state()
        x_T = self.x_t
        l_T = self.l_t
        sigma_x = self.sigmas[-1]
        adjacent_sigma_x = self.sigmas[-2]

        # Hard coded predictor step
        pred_l, pred_x = self.decoder(None, None, 0, 0, None, None)
        pred_x *= torch.sqrt(self.sigma_norm)
        rand_x = torch.randn_like(x_T)
        rand_l = torch.randn_like(l_T)
        step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
        std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   
        new_x = self.x_t - step_size * pred_x + std_x * rand_x

        # Call wrapper
        batch = Batch(None, 1)
        torch.set_rng_state(state)
        wrap_x, _ = self.predictor_wrapper.get_predictor_update(0, self.x_t, self.l_t, batch, self.sigma_norm)

        # Assert equality
        assert torch.all(torch.eq(new_x, wrap_x))

if __name__ == '__main__':
    unittest.main()
