# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion for discrete state spaces."""

import diffcsp.d3pm_utils
# import jax
# from jax import lax
# import jax.numpy as jnp
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import scipy.special
from loguru import logger


# TODO: Modify transition matrices other than absorbing to work with atom diffusion
#### At the moment only absorbing is modified to work 

# TODO: Modify how loss is computed so we can more easily join with DiffCSP

def make_diffusion(hps):
    """HParams -> diffusion object."""
    return CategoricalDiffusion(
        betas=get_diffusion_betas(hps.diffusion_betas),
        model_prediction=hps.model_prediction,
        model_output=hps.args.model_output,
        transition_mat_type=hps.transition_mat_type,
        transition_bands=hps.transition_bands,
        loss_type=hps.loss_type,
        hybrid_coeff=hps.hybrid_coeff,
        num_classes=hps.args.num_classes)


def get_diffusion_betas(spec):
    """Get betas from the hyperparameters."""
    if spec.type == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return torch.linspace(spec.start, spec.stop, spec.num_timesteps)
    elif spec.type == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = (
                np.arange(spec.num_timesteps + 1, dtype=np.float64) /
                spec.num_timesteps)
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = torch.from_numpy(np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999))
        return betas
    elif spec.type == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / torch.linspace(spec.num_timesteps, 1., spec.num_timesteps)
    else:
        raise NotImplementedError(spec.type)


class CategoricalDiffusion(nn.Module):
    """Discrete state space diffusion process.

    Time convention: noisy data is labeled x_0, ..., x_{T-1}, and original data
    is labeled x_start (or x_{-1}). This convention differs from the papers,
    which use x_1, ..., x_T for noisy data and x_0 for original data.
    """

    def __init__(self, *, betas, model_prediction, model_output,
                 transition_mat_type, transition_bands, loss_type, hybrid_coeff,
                 num_classes):
        super().__init__()

        self.model_prediction = model_prediction  # x_start, xprev
        self.model_output = model_output  # logits or logistic_pars
        self.loss_type = loss_type  # kl, hybrid, cross_entropy_x_start
        self.hybrid_coeff = hybrid_coeff

        # Data \in {0, ..., num_classes-1}
        self.num_classes = num_classes
        self.transition_bands = transition_bands
        self.transition_mat_type = transition_mat_type
        self.eps = 1.e-6

        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError('betas must be in (0, 1]')

        # Computations here in float64 for accuracy
        # self.betas = betas
        self.register("betas", betas)
        self.num_timesteps, = betas.shape

        # Construct transition matrices for q(x_t|x_{t-1})
        # NOTE: t goes from {0, ..., T-1}
        logger.info('[compute transition matrix]: {}', self.transition_mat_type)
        if self.transition_mat_type == 'uniform':
            q_one_step_mats = [self._get_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'gaussian':
            q_one_step_mats = [self._get_gaussian_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'absorbing':
            q_one_step_mats = [self._get_absorbing_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
            )
        self.register("q_onestep_mats", torch.stack(q_one_step_mats, dim=0))
        # self.q_onestep_mats = torch.stack(q_one_step_mats, dim=0)
        assert self.q_onestep_mats.shape == (self.num_timesteps,
                                             self.num_classes,
                                             self.num_classes)
        logger.info('[trainsition matrix]: {}', self.q_onestep_mats.shape)

        # Construct transition matrices for q(x_t|x_start)
        logger.info('[Construct transition matrices for q(x_t|x_start)]')
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t],
                                      dims=[[1], [0]])
            q_mats.append(q_mat_t)
        # self.q_mats = torch.stack(q_mats, dim=0)
        self.register("q_mats", torch.stack(q_mats, dim=0))
        assert self.q_mats.shape == (self.num_timesteps, self.num_classes,
                                     self.num_classes), self.q_mats.shape
        logger.info('[tilde(Q)t]: {}', self.q_mats.shape)

        # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
        # Can be computed from self.q_mats and self.q_one_step_mats.
        # Only need transpose of q_onestep_mats for posterior computation.
        # self.transpose_q_onestep_mats = torch.transpose(self.q_onestep_mats,1,2)
        self.register("transpose_q_onestep_mats", torch.transpose(self.q_onestep_mats, 1, 2))
        # del self.q_onestep_mats

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def _get_full_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Contrary to the band diagonal version, this method constructs a transition
        matrix with uniform probability to all other states.

        Args:
          t: timestep. integer scalar.

        Returns:
          Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t].numpy()
        mat = np.full(shape=(self.num_classes, self.num_classes),
                      fill_value=beta_t / float(self.num_classes),
                      dtype=np.float64)
        diag_indices = np.diag_indices_from(mat)
        diag_val = 1. - beta_t * (self.num_classes - 1.) / self.num_classes
        mat[diag_indices] = diag_val
        return torch.from_numpy(mat)

    def _get_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

        This method constructs a transition
        matrix Q with
        Q_{ij} = beta_t / num_classes       if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il} if i==j.
                 0                          else.

        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)
        # Assumes num_off_diags < num_classes
        beta_t = self.betas[t].numpy()

        mat = np.zeros((self.num_classes, self.num_classes),
                       dtype=np.float64)
        off_diag = np.full(shape=(self.num_classes - 1,),
                           fill_value=beta_t / float(self.num_classes),
                           dtype=np.float64)
        for k in range(1, self.transition_bands + 1):
            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)
        return torch.from_numpy(mat)

    def _get_gaussian_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

        This method constructs a transition matrix Q with
        decaying entries as a function of how far off diagonal the entry is.
        Normalization option 1:
        Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il}  if i==j.
                 0                          else.

        Normalization option 2:
        tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                         0                        else.

        Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        transition_bands = self.transition_bands if self.transition_bands else self.num_classes - 1

        beta_t = self.betas[t].numpy()

        mat = np.zeros((self.num_classes, self.num_classes),
                       dtype=np.float64)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        values = np.linspace(start=0., stop=255., num=self.num_classes,
                             endpoint=True, dtype=np.float64)
        values = values * 2. / (self.num_classes - 1.)
        values = values[:transition_bands + 1]
        values = -values * values / beta_t

        values = np.concatenate([values[:0:-1], values], axis=0)
        values = scipy.special.softmax(values, axis=0)
        values = values[transition_bands:]
        for k in range(1, transition_bands + 1):
            off_diag = np.full(shape=(self.num_classes - k,),
                               fill_value=values[k],
                               dtype=np.float64)

            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)

        return torch.from_numpy(mat)


    def _get_absorbing_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Has an absorbing state for pixelvalues self.num_pixel_vals//2.

        Args:
          t: timestep. integer scalar.

        Returns:
          Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
        """
        beta_t = self.betas[t].numpy()

        diag = np.full(shape=(self.num_classes,), fill_value=1. - beta_t,
                       dtype=np.float64)
        mat = np.diag(diag, k=0)
        # mask state is 0th indexed class
        mat[:, 0] += beta_t

        return torch.from_numpy(mat)

    def _at(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: np.ndarray: plain NumPy float64 array of constants indexed by time.
          t: jnp.ndarray: Jax array of time indices, shape = (batch_size,).
          x: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one hot representation, but have integer
            values representing the class values.

        Returns:
          a[t, x]: jnp.ndarray: Jax array.
        """
        # a = np.asarray(a, dtype=self.jax_dtype)
        # t_broadcast = np.expand_dims(t, tuple(range(1, x.ndim))).tolist()

        # x.shape = (bs, channels, height, width)
        # t.shape = (bs)
        # a.shape = (num_timesteps, num_classes, num_classes)
        # out.shape = (bs, channels, height, width, num_classes)
        # out[i, j, k, l, m] = a[t[i], x[i, j, k, l], m]
        B, C = x.shape
        a_t = torch.index_select(a, dim=0, index=t)
        assert a_t.shape == (B, self.num_classes, self.num_classes)
        # out = a_t[x.tolist()]
        x_onehot = F.one_hot(x.view(B, -1).to(torch.int64), num_classes=self.num_classes).to(torch.float32)
        out = torch.matmul(x_onehot, a_t)
        out = out.view(B, C, self.num_classes)
        return out

    def _at_eric(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: np.ndarray: plain NumPy float64 array of constants indexed by time.
          t: jnp.ndarray: Jax array of time indices, shape = (batch_size,).
          x: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one hot representation, but have integer
            values representing the class values.

        Returns:
          a[t, x]: jnp.ndarray: Jax array.
        """
        # a = np.asarray(a, dtype=self.jax_dtype)
        # t_broadcast = np.expand_dims(t, tuple(range(1, x.ndim))).tolist()

        # x.shape = (bs, channels, height, width)
        # t.shape = (bs)
        # a.shape = (num_timesteps, num_classes, num_classes)
        # out.shape = (bs, channels, height, width, num_classes)
        # out[i, j, k, l, m] = a[t[i], x[i, j, k, l], m]
        B = x.shape
        a_t = torch.index_select(a, dim=0, index=t)
        #print (a_t.shape)
        #assert a_t.shape == (B, self.num_classes, self.num_classes)
        # out = a_t[x.tolist()]
        x_onehot = F.one_hot(x.view(B, -1).to(torch.int64), num_classes=self.num_classes).to(torch.float32)
        #print (x_onehot.shape)
        out = torch.matmul(x_onehot, a_t)
        #print(out.shape)
        return out

    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: np.ndarray: plain NumPy float64 array of constants indexed by time.
          t: jnp.ndarray: Jax array of time indices, shape = (bs,).
          x: jnp.ndarray: jax array, shape (bs, ..., num_classes), float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
          out: jnp.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, ..., num_classes)
        """

        # x.shape = (bs, channels, height, width, num_classes)
        # a[t]shape = (bs, num_classes, num_classes)
        # out.shape = (bs, height, width, channels, num_classes)
        B, C, W = x.shape
        a_t = torch.index_select(a, dim=0, index=t)
        assert a_t.shape == (x.shape[0], self.num_classes, self.num_classes)
        x = x.view(B, -1, self.num_classes)
        out = torch.matmul(x, a_t)
        out = out.view(B, C, self.num_classes)
        return out

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
          x_start: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
             Should not be of one hot representation, but have integer values
             representing the class values.
          t: jnp.ndarray: jax array of shape (bs,).

        Returns:
          probs: jnp.ndarray: jax array, shape (bs, x_start.shape[1:],
                                                num_classes).
        """
        return self._at(self.q_mats, t, x_start)

    def q_probs_eric(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
          x_start: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
             Should not be of one hot representation, but have integer values
             representing the class values.
          t: jnp.ndarray: jax array of shape (bs,).

        Returns:
          probs: jnp.ndarray: jax array, shape (bs, x_start.shape[1:],
                                                num_classes).
        """
        return self._at_eric(self.q_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).

        Args:
          x_start: jnp.array: original clean data, in integer form (not onehot).
            shape = (bs, ...).
          t: :jnp.array: timestep of the diffusion process, shape (bs,).
          noise: jnp.ndarray: uniform noise on [0, 1) used to sample noisy data.
            Should be of shape (*x_start.shape, num_classes).

        Returns:
          sample: jnp.ndarray: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.num_classes,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = - torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_sample_eric(self, x_start, t, noise):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).

        Args:
          x_start: jnp.array: original clean data, in integer form (not onehot).
            shape = (bs, ...).
          t: :jnp.array: timestep of the diffusion process, shape (bs,).
          noise: jnp.ndarray: uniform noise on [0, 1) used to sample noisy data.
            Should be of shape (*x_start.shape, num_classes).

        Returns:
          sample: jnp.ndarray: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.num_classes,)
        logits = torch.log(self.q_probs_eric(x_start, t) + self.eps)

        # To avoid numerical issues clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = - torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def _get_logits_from_logistic_pars(self, loc, log_scale):
        """Computes logits for an underlying logistic distribution."""

        loc = torch.unsqueeze(loc, dim=-1)
        log_scale = torch.unsqueeze(log_scale, dim=-1)

        # Shift log_scale such that if it's zero the probs have a scale
        # that is not too wide and not too narrow either.
        inv_scale = torch.exp(- (log_scale - 2.))

        bin_width = 2. / (self.num_classes - 1.)
        bin_centers = torch.linspace(start=-1., end=1., steps=self.num_classes)

        bin_centers = torch.tensor(np.expand_dims(bin_centers.numpy(),
                                                  axis=tuple(range(0, loc.ndim - 1)))).to(loc.device)

        bin_centers = bin_centers - loc
        log_cdf_min = F.logsigmoid(
            inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = F.logsigmoid(
            inv_scale * (bin_centers + 0.5 * bin_width))

        logits = d3pm_utils.log_min_exp(log_cdf_plus, log_cdf_min, self.eps)

        # Normalization:
        # # Option 1:
        # # Assign cdf over range (-\inf, x + 0.5] to pmf for pixel with
        # # value x = 0.
        # logits = logits.at[..., 0].set(log_cdf_plus[..., 0])
        # # Assign cdf over range (x - 0.5, \inf) to pmf for pixel with
        # # value x = 255.
        # log_one_minus_cdf_min = - jax.nn.softplus(
        #     inv_scale * (bin_centers - 0.5 * bin_width))
        # logits = logits.at[..., -1].set(log_one_minus_cdf_min[..., -1])
        # # Option 2:
        # # Alternatively normalize by reweighting all terms. This avoids
        # # sharp peaks at 0 and 255.
        # since we are outputting logits here, we don't need to do anything.
        # they will be normalized by softmax anyway.

        return logits

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start)."""

        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_classes,), (
                x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            t_1 = torch.where(t == 0, t, t - 1)
            fact2 = self._at_onehot(self.q_mats, t_1,
                                    F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            t_1 = torch.where(t == 0, t, t-1)
            fact2 = self._at(self.q_mats, t_1, x_start)
            tzero_logits = torch.log(
                F.one_hot(x_start.to(torch.int64), num_classes=self.num_classes)
                + self.eps)

        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        # t_broadcast = np.expand_dims(t, tuple(range(1, out.ndim)))
        t_broadcast = torch.reshape(t, ([out.shape[0]] + [1] * (len(out.shape) - 1)))
        return torch.where(t_broadcast == 0, tzero_logits,
                           out)
    def q_posterior_logits_eric(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start)."""
        
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_classes,), (
                x_start.shape, x_t.shape)
        else: 
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)
        
        fact1 = self._at_eric(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            t_1 = torch.where(t == 0, t, t - 1)
            fact2 = self._at_onehot(self.q_mats, t_1,
                                    F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            t_1 = torch.where(t == 0, t, t-1)
            fact2 = self._at_eric(self.q_mats, t_1, x_start)
            tzero_logits = torch.log(
                F.one_hot(x_start.to(torch.int64), num_classes=self.num_classes)
                + self.eps)

        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        # t_broadcast = np.expand_dims(t, tuple(range(1, out.ndim)))
        t_broadcast = torch.reshape(t, ([out.shape[0]] + [1] * (len(out.shape) - 1)))
        return torch.where(t_broadcast == 0, tzero_logits,
                           out)

    def p_logits(self, model_fn, *, x, t):
        """Compute logits of p(x_{t-1} | x_t)."""
        assert t.shape == (x.shape[0],)
        model_output = model_fn(x, t)
        # model_output = torch.full(model_output.shape, 0.039, dtype=model_output.dtype).to(model_output.device)

        if self.model_output == 'logits':
            model_logits = model_output

        elif self.model_output == 'logistic_pars':
            # Get logits out of discretized logistic distribution.
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)

        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == 'x_start':
            # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
            # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
            pred_x_start_logits = model_logits

            # t_broadcast = np.expand_dims(t, tuple(range(1, model_logits.ndim)))
            t_broadcast = torch.reshape(t, ([model_logits.shape[0]] + [1] * (len(model_logits.shape) - 1)))
            model_logits = torch.where(t_broadcast == 0,
                                       pred_x_start_logits,
                                       self.q_posterior_logits(pred_x_start_logits, x,
                                                               t, x_start_logits=True)
                                       )

        elif self.model_prediction == 'xprev':
            # Use the logits out of the model directly as the logits for
            # p(x_{t-1}|x_t). model_logits are already set correctly.
            # NOTE: the pred_x_start_logits in this case makes no sense.
            # For Gaussian DDPM diffusion the model predicts the mean of
            # p(x_{t-1}}|x_t), and uses inserts this as the eq for the mean of
            # q(x_{t-1}}|x_t, x_0) to compute the predicted x_0/x_start.
            # The equivalent for the categorical case is nontrivial.
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)

        assert (model_logits.shape ==
                pred_x_start_logits.shape == x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits

    def standalone_p_logits(self, output_logits, *, x, t):
        """Compute logits of p(x_{t-1} | x_t)."""
        assert t.shape == (x.shape[0],)
            
        if self.model_output == 'logits':
            model_logits = output_logits
        
        elif self.model_output == 'logistic_pars':
            # Get logits out of discretized logistic distribution.
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
            
        else:
            raise NotImplementedError(self.model_output)
            
        if self.model_prediction == 'x_start':
            # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
            # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
            pred_x_start_logits = model_logits
        
            # t_broadcast = np.expand_dims(t, tuple(range(1, model_logits.ndim)))
            t_broadcast = torch.reshape(t, ([model_logits.shape[0]] + [1] * (len(model_logits.shape) - 1)))
            model_logits = torch.where(t_broadcast == 0,
                                       pred_x_start_logits,
                                       self.q_posterior_logits_eric(pred_x_start_logits, x,
                                                               t, x_start_logits=True)
                                       )

        elif self.model_prediction == 'xprev':
            # Use the logits out of the model directly as the logits for
            # p(x_{t-1}|x_t). model_logits are already set correctly.
            # NOTE: the pred_x_start_logits in this case makes no sense.
            # For Gaussian DDPM diffusion the model predicts the mean of
            # p(x_{t-1}}|x_t), and uses inserts this as the eq for the mean of
            # q(x_{t-1}}|x_t, x_0) to compute the predicted x_0/x_start.
            # The equivalent for the categorical case is nontrivial.
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)

        assert (model_logits.shape ==
                pred_x_start_logits.shape == x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits

    # === Sampling ===
    @torch.no_grad()
    def p_sample(self, model_fn, *, x, t, noise):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        model_logits, pred_x_start_logits = self.p_logits(
            model_fn=model_fn, x=x, t=t)
        assert noise.shape == model_logits.shape, noise.shape

        # No noise when t == 0
        # NOTE: for t=0 this just "samples" from the argmax
        #   as opposed to "sampling" from the mean in the gaussian case.
        nonzero_mask = (t != 0).to(x.dtype).reshape(x.shape[0],
                                                        *([1] * (len(x.shape))))
        # For numerical precision clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        return sample, F.softmax(pred_x_start_logits, dim=-1)

    @torch.no_grad()
    def p_sample_loop(self, model_fn, shape,
                      num_timesteps=None, return_x_init=False):
        """Ancestral sampling."""
        # init_rng, body_rng = jax.random.split(rng)
        # del rng

        device = 'cuda' if next(model_fn.parameters()).is_cuda else 'cpu'
        logger.info(device)
        noise_shape = shape + (self.num_classes,)

        if self.transition_mat_type in ['gaussian', 'uniform']:
            # Stationary distribution is a uniform distribution over all pixel values.
            x_init = torch.randint(size=shape, low=0, high=self.num_classes).to(device)

        elif self.transition_mat_type == 'absorbing':
            # Stationary distribution is a kronecker delta distribution
            # with all its mass on the absorbing state.
            # Absorbing state is located at 0th class
            x_init = torch.full(size=shape, fill_value=0,
                                dtype=torch.int32).to(device)
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
            )

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        x = x_init
        for i in reversed(range(0, num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.int64).to(device)
            x, _ = self.p_sample(
                model_fn=model_fn,
                x=x,
                t=t,
                noise=torch.rand(size=noise_shape).to(x.device)
            )
            print (x[:5,:])

        assert x.shape == shape
        if return_x_init:
            return x_init, x
        else:
            return x

    # === Log likelihood / loss calculation ===

    def vb_terms_bpd(self, model_fn, *, x_start, x_t, t):
        """Calculate specified terms of the variational bound.

        Args:
          model_fn: the denoising network
          x_start: original clean data
          x_t: noisy data
          t: timestep of the noisy data (and the corresponding term of the bound
            to return)

        Returns:
          a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
          (specified by `t`), and `pred_x_start_logits` is logits of
          the denoised image.
        """
        batch_size = t.shape[0]
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t)

        kl = d3pm_utils.categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        assert kl.shape == x_start.shape
        kl = d3pm_utils.meanflat(kl) / np.log(2.)
        #kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)

        decoder_nll = -d3pm_utils.categorical_log_likelihood(x_start, model_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = d3pm_utils.meanflat(decoder_nll) / np.log(2.)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        return torch.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def standalone_vb_terms_bpd(self, output_logits, *, x_start, x_t, t):
        """Calculate specified terms of the variational bound.

        Args:
          output_logits: model output
          x_start: original clean data
          x_t: noisy data
          t: timestep of the noisy data (and the corresponding term of the bound
            to return)

        Returns:
          a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
          (specified by `t`), and `pred_x_start_logits` is logits of
          the denoised image.
        """
        batch_size = t.shape[0]
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.standalone_p_logits(output_logits, x=x_t, t=t)

        kl = d3pm_utils.categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        assert kl.shape == x_start.shape
        kl = d3pm_utils.meanflat(kl) / np.log(2.)
        #kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)

        decoder_nll = -d3pm_utils.categorical_log_likelihood(x_start, model_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = d3pm_utils.meanflat(decoder_nll) / np.log(2.)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        return torch.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def prior_bpd(self, x_start):
        """KL(q(x_{T-1}|x_start)|| U(x_{T-1}|0, num_classes-1))."""
        q_probs = self.q_probs(
            x_start=x_start,
            t=torch.full((x_start.shape[0],), self.num_timesteps - 1).to(x_start.device)
        )

        if self.transition_mat_type in ['gaussian', 'uniform']:
            # Stationary distribution is a uniform distribution over all pixel values.
            prior_probs = torch.ones_like(q_probs) / self.num_classes

        elif self.transition_mat_type == 'absorbing':
            # Stationary distribution is a kronecker delta distribution
            # with all its mass on the absorbing state.
            # Absorbing state is located at rgb values (128, 128, 128)
            absorbing_int = torch.full(size=q_probs.shape[:-1],
                                       fill_value=0.,
                                       dtype=torch.int32)
            prior_probs = F.one_hot(absorbing_int.to(torch.int64),
                                    num_classes=self.num_classes)
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
            )

        assert prior_probs.shape == q_probs.shape

        kl_prior = d3pm_utils.categorical_kl_probs(
            q_probs, prior_probs)
        assert kl_prior.shape == x_start.shape
        return d3pm_utils.meanflat(kl_prior) / np.log(2.)

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """Calculate crossentropy between x_start and predicted x_start.

        Args:
          x_start: original clean data
          pred_x_start_logits: predicted_logits

        Returns:
          ce: cross entropy.
        """

        ce = -d3pm_utils.categorical_log_likelihood(x_start, pred_x_start_logits)
        assert ce.shape == x_start.shape
        ce = d3pm_utils.meanflat(ce) / np.log(2.)

        assert ce.shape == (x_start.shape[0],)

        return ce

    def training_losses(self, model_fn, x_start, t,noise, val=False):
        """Training loss calculation."""

        # Add noise to data
        # noise_rng, time_rng = jax.random.split(rng)
        #noise = torch.rand(size=x_start.shape + (self.num_classes,)).to(x_start.device)

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
        # itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Calculate the loss
        if self.loss_type == 'kl':
            # Optimizes the variational bound L_vb.
            losses, _ = self.vb_terms_bpd(
                model_fn=model_fn, x_start=x_start, x_t=x_t, t=t)

        elif self.loss_type == 'cross_entropy_x_start':
            # Optimizes - sum_x_start x_start log pred_x_start.
            _, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t)
            losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logits)

        elif self.loss_type == 'hybrid':
            # Optimizes L_vb - lambda * sum_x_start x_start log pred_x_start.
            vb_losses, pred_x_start_logits = self.vb_terms_bpd(
                model_fn=model_fn, x_start=x_start, x_t=x_t, t=t)
            ce_losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            losses = vb_losses + self.hybrid_coeff * ce_losses
            #if val:
            #    class_pred = torch.argmax(pred_x_start_logits,dim=-1)
            #    for i in range (x_start.shape[0]):
            #        print (x_start[i,:],class_pred[i,:])

        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == t.shape
        return losses

    def standalone_losses(self, x_start, t, x_t, output_logits, val=False):
        """Use within DiffCSP """
    
        # Mpve noise outside of loss computation
        #noise = torch.rand(size=x_start.shape + (self.num_classes,)).to(x_start.device)
                                       
        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
        # itself.                                              
        #x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Calculate the loss
        if self.loss_type == 'kl':
            # Optimizes the variational bound L_vb.
            losses, _ = self.standalone_vb_terms_bpd(
                output_logits=output_logits, x_start=x_start, x_t=x_t, t=t)

        elif self.loss_type == 'cross_entropy_x_start':
            # Optimizes - sum_x_start x_start log pred_x_start.
            _, pred_x_start_logits = self.standalone_p_logits(output_logits, x=x_t, t=t)
            losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logits)

        elif self.loss_type == 'hybrid':
            # Optimizes L_vb - lambda * sum_x_start x_start log pred_x_start.
            vb_losses, pred_x_start_logits = self.standalone_vb_terms_bpd(
                output_logits=output_logits, x_start=x_start, x_t=x_t, t=t)
            ce_losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            losses = vb_losses + self.hybrid_coeff * ce_losses
            if val:
                class_pred = torch.argmax(pred_x_start_logits,dim=-1)
                for i in range (x_start.shape[0]):
                    print (x_start[i,:],class_pred[i,:])
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == t.shape
        return losses



    def get_noisy_data(self, x_start, t):
        """Get noisy data without training loss calculation."""

        # Add noise to data
        # noise_rng, time_rng = jax.random.split(rng)
        noise = torch.rand(size=x_start.shape + (self.num_classes,)).to(x_start.device)

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
        # itself.
        x_t = self.q_sample_eric(x_start=x_start, t=t, noise=noise)
        return x_t

    @torch.no_grad()
    def calc_bpd_loop(self, model_fn, x_start):
        """Calculate variational bound (loop over all timesteps and sum)."""
        batch_size = x_start.shape[0]

        noise_shape = x_start.shape + (self.num_classes,)

        vbterms = []
        for t in reversed(range(self.num_timesteps)):
            t_b = torch.full((batch_size,), t).to(x_start.device)

            vb, _ = self.vb_terms_bpd(
                model_fn=model_fn, x_start=x_start, t=t_b,
                x_t=self.q_sample(
                    x_start=x_start, t=t_b,
                    noise=torch.rand(size=noise_shape).to(x_start.device)
                ))

            vbterms.append(vb)

        vbterms_tb = torch.stack(vbterms, dim=0)
        vbterms_bt = vbterms_tb.T
        assert vbterms_bt.shape == (batch_size, self.num_timesteps)

        prior_b = self.prior_bpd(x_start=x_start)
        total_b = vbterms_tb.sum(dim=0) + prior_b
        assert prior_b.shape == total_b.shape == (batch_size,)

        return {
            'total': total_b,
            'vbterms': vbterms_bt,
            'prior': prior_b,
        }

if __name__ == "__main__":


    import ml_collections


    def config_dict(**kwargs):
        return ml_collections.ConfigDict(initial_dictionary=kwargs)
    
    cd = config_dict(diffusion_betas=config_dict(
                type='jsd',
                # start, stop only relevant for linear, power, jsdtrunc schedules.
                start=1e-4,  # 1e-4 gauss, 0.02 uniform, not needed for jsd
                stop=0.02,  # 0.02, gauss, 1. uniform, not needed for jsd
                num_timesteps=1000,
            ),
            # Settings used in diffusion_categorical.py
            model_prediction='x_start',  # 'x_start','xprev'
            # 'gaussian','uniform','absorbing'
            transition_mat_type='absorbing',
            transition_bands=None,
            loss_type='hybrid',  # kl,cross_entropy_x_start, hybrid
            hybrid_coeff=0.001,  # only used for hybrid loss type.
            args=config_dict(num_classes=4,
                model_output="logits"),
        )

    dd = make_diffusion(cd)

    x = torch.randint(1,4,(100,))
    print (x.shape)
    for t in range (1000):
        print (t,torch.count_nonzero(dd.get_noisy_data(x,torch.tensor([t]))).item())
