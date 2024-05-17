import os
import sys
import torch
sys.path.append(os.path.abspath("../../score_sde_pytorch"))
import score_sde_pytorch.sampling as sampling
from score_sde_pytorch import sde_lib
sys.path.pop()


class CorrectorWrapper(object):
    # TODO: This should probably be set up with hydra?
    def __init__(self, corrector_name_x: str, corrector_name_l: str,
                 sde_name_x: str, sde_name_l: str, decoder,
                 number_corrector_steps: int = 1) -> None:
        corrector_class_x = sampling.get_corrector(corrector_name_x.lower())
        corrector_class_l = sampling.get_corrector(corrector_name_l.lower())


        # Annealed Langevin corrector uses sde.alphas (depending on beta range)
        # and sde.marginal_prob, and sde.N, and sde.T
        # Langevin corrector uses N, T, alphas.
        if sde_name_x.lower() == "vpsde":
            sde_x = sde_lib.VPSDE(beta_min=..., beta_max=..., N=...)
        elif sde_name_x.lower() == "subvpsde":
            sde_x = sde_lib.subVPSDE(beta_min=..., beta_max=..., N=...)
        elif sde_name_x.lower() == "vesde":
            sde_x = sde_lib.VESDE(sigma_min=..., sigma_max=..., N=...)
        else:
            raise ValueError(f"Unknown sde name: {sde_name_x}")

        if sde_name_l.lower() == "vpsde":
            sde_l = sde_lib.VPSDE(beta_min=..., beta_max=..., N=...)
        elif sde_name_l.lower() == "subvpsde":
            sde_l = sde_lib.subVPSDE(beta_min=..., beta_max=..., N=...)
        elif sde_name_l.lower() == "vesde":
            sde_l = sde_lib.VESDE(sigma_min=..., sigma_max=..., N=...)
        else:
            raise ValueError(f"Unknown sde name: {sde_name_l}")

        self._score_x = None
        self._score_l = None
        # TODO: Set up the sde and snr arguments correctly.
        self._corrector_x = corrector_class_x(sde=sde_x, score_fn=lambda x, t: self.get_score_x(), snr=None, n_steps=1)
        self._corrector_l = corrector_class_l(sde=sde_l, score_fn=lambda l, t: self.get_score_l(), snr=None, n_steps=1)
        self._decoder = decoder
        self._number_corrector_steps = number_corrector_steps

    def get_corrector_update(self, time_emb: torch.Tensor, atom_types: torch.tensor, x_t: torch.tensor,
                             l_t: torch.tensor, num_atoms: torch.tensor, batch: torch.tensor):
        for i in range(self._number_corrector_steps):
            self._score_l, self._score_x = self._decoder(time_emb, atom_types, x_t, l_t, num_atoms, batch)
            x_t, _ = self._corrector_x.update_fn(x_t, time_emb)
            l_t, _ = self._corrector_l.update_fn(l_t, time_emb)
        return x_t, l_t

    def get_score_x(self):
        # self._score_x is set in get_corrector_update after calling the decoder.
        # We make sure that each score is only used once by setting it to None returning it here.
        assert self._score_x is not None
        score = self._score_x
        self._score_x = None
        return score

    def get_score_l(self):
        # Same trick as for self._score_x.
        assert self._score_l is not None
        score = self._score_l
        self._score_l = None
        return score
