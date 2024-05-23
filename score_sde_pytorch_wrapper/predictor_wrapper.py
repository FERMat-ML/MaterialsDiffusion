import os
import sys
import torch
sys.path.append(os.path.abspath("../score_sde_pytorch"))
import score_sde_pytorch.sampling as sampling
sys.path.pop()

class PredictorWrapper(object):

    def __init__(self, predictor_name_x:str, predictor_name_l:str, sde_name_x:str, sde_name_l:str,
                 decoder:torch.nn.Module, step_lr_x:float, step_lr_l:float, sigma_begin_x:torch.Tensor,
                 number_predictor_steps:int=1) -> None:

        predictor_class_x = sampling.get_predictor(predictor_name_x.lower())
        predictor_class_l = sampling.get_predictor(predictor_name_l.lower())

        # Set SDE for fractional coords.
        if sde_name_x.lower() == "vpsde":
            sde_x = sampling.sde_lib.VPSDE()
        elif sde_name_x.lower() == "subvpsde":
            sde_x = sampling.sde_lib.subVPSDE()
        elif sde_name_x.lower() == "vesde":
            sde_x = sampling.sde_lib.VESDE()
        else:
            raise ValueError(f"Unknown sde name: {sde_name_x}")

        # Set SDE for lattice
        if sde_name_l.lower() == "vpsde":
            sde_l = sampling.sde_lib.VPSDE()
        elif sde_name_l.lower() == "subvpsde":
            sde_l = sampling.sde_lib.subVPSDE()
        elif sde_name_l.lower() == "vesde":
            sde_l = sampling.sde_lib.VESDE()
        else:
            raise ValueError(f"Unknown sde name: {sde_name_l}")

        self.score_x = None
        self.score_l = None

        # Set predictors
        self.predictor_x = predictor_class_x(sde=sde_x, score_fn=lambda x,t: self._get_score_x())
        self.predictor_l = predictor_class_l(sde=sde_l, score_fn=lambda x,t: self._get_score_l())

        # Other variables
        self._decoder = decoder
        self.dummy_tensor = torch.tensor([1])

    def get_predictor_update(self, time_emb:torch.Tensor, x_t:torch.tensor, l_t:torch.tensor,
                             l_t:torch.tensor, num_atoms:torch.tensor, batch:torch.tensor, sigma_norm:torch.tensor,
                             sigma_x:torch.tensor,) -> (torch.Tensor, torch.Tensor):

        '''
        Function to run predictor and update positions and fractional coordinates

        @param time_emb : time embedding
        @param x_t : fractional coordinates at time t in sampling process
        @param l_t : lattice at time t in sampling process
        @param num_atoms : number of atoms
        @param batch : current batch
        @param sigma_norm : norm of sigma at time t
        @param sigma_x : sigma for fractional coordinates at time t
        '''

        # Iterate over number of steps
        for i in range(len(self._number_predictor_steps)):

            # Get score
            self._score_l, self._score_x = self._decoder(time_emb, batch.atom_types, x_t, x_l, num_atoms, batch)
            self._score_x *= -torch.sqrt(sigma_norm)

            # Update
            x_t, _ = self.predictor_x.update_fn(x_t, self.dummy_tensor)
            l_t, _ = self.predictor_l.update_fn(l_t, self.dummy_tensor)

        # Return
        return x_t, l_t

    def _get_score_x(self):
        '''
        Get score and clear score for next iteration

        @return score : model score for fractional coordinates at iteration
        '''

        # Get and return score
        assert self._score_x is not None
        score = self._score_x
        self._score_x = None
        return score

    def _get_score_l(self):
        '''
        Get score and clear score for next iteration

        @return score : model score for lattice at iteration
        '''

        # Get and return score
        assert self._score_l is not None
        score = self._score_l
        self._score_l = None
        return score