import os
import sys
import torch
# TODO: Set up git dependency so that everybody has this repository in the lib directory.
sys.path.append(os.path.abspath("../../lib/score_sde_pytorch"))
import lib.score_sde_pytorch.sampling as sampling
sys.path.pop()


class CorrectorWrapper(object):
    def __init__(self, corrector_name_x: str, corrector_name_l: str, decoder,
                 number_corrector_steps: int = 1) -> None:
        corrector_class_x = sampling.get_corrector(corrector_name_x)
        corrector_class_l = sampling.get_corrector(corrector_name_l)

        self._score_x = None
        self._score_l = None
        # TODO: Set up the sde and snr arguments correctly.
        self._corrector_x = corrector_class_x(sde=None, score_fn=lambda x, t: self.get_score_x(), snr=None, n_steps=1)
        self._corrector_l = corrector_class_l(sde=None, score_fn=lambda l, t: self.get_score_l(), snr=None, n_steps=1)
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
