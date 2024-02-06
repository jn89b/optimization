""" Gaussian Process custom implementation for the data-augmented MPC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
import casadi as cs
import joblib

from tqdm import tqdm
from operator import itemgetter
from numpy.linalg import inv, cholesky, lstsq
from numpy.random import mtrand
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform

# from src.utils.utils import safe_mknode_recursive, make_bz_matrix


class CustomKernelFunctions:

    def __init__(self, kernel_func, params=None):

        self.params = params
        self.kernel_type = kernel_func

        if self.kernel_type == 'squared_exponential':
            if params is None:
                self.params = {'l': [1.0], 'sigma_f': 1.0}
            self.kernel = self.squared_exponential_kernel
        else:
            raise NotImplementedError("only squared_exponential is supported")

        self.theta = self.get_trainable_parameters()

    def __call__(self, x_1, x_2):
        return self.kernel(x_1, x_2)

    def __str__(self):
        if self.kernel_type == 'squared_exponential':
            len_scales = np.reshape(self.params['l'], -1)
            len_scale_str = '['
            for i in range(len(len_scales)):
                len_scale_str += '%.3f, ' % len_scales[i] if i < len(len_scales) - 1 else '%.3f' % len_scales[i]
            len_scale_str += ']'
            summary = '%.3f' % self.params['sigma_f']
            summary += '**2*RBF(length_scale=' + len_scale_str + ')'
            return summary

        else:
            raise NotImplementedError("only squared_exponential is supported")

    def get_trainable_parameters(self):
        trainable_params = []
        if self.kernel_type == 'squared_exponential':
            trainable_params += \
                np.reshape(np.squeeze(self.params['l']), -1).tolist() if hasattr(self.params['l'], "__len__") \
                else [self.params['l']]
            trainable_params += [self.params['sigma_f']]
        return trainable_params

    @staticmethod
    def _check_length_scale(x, length_scale):
        length_scale = np.squeeze(length_scale).astype(float)
        if np.ndim(length_scale) > 1:
            raise ValueError("length_scale cannot be of dimension greater than 1")
        if np.ndim(length_scale) == 1 and x.shape[1] != length_scale.shape[0]:
            raise ValueError("Anisotropic kernel must have the same number of dimensions as data (%d!=%d)"
                             % (length_scale.shape[0], x.shape[1]))
        return length_scale

    def squared_exponential_kernel(self, x_1, x_2=None):
        """
        Anisotropic (diagonal length-scale) matrix squared exponential kernel. Computes a covariance matrix from points
        in x_1 and x_2.

        Args:
            x_1: Array of m points (m x d).
            x_2: Array of n points (n x d).

        Returns:
            Covariance matrix (m x n).
        """

        if isinstance(x_2, cs.MX):
            return self._squared_exponential_kernel_cs(x_1, x_2)

        # Length scale parameter
        len_scale = self.params['l'] if 'l' in self.params.keys() else 1.0

        # Vertical variation parameter
        sigma_f = self.params['sigma_f'] if 'sigma_f' in self.params.keys() else 1.0

        x_1 = np.atleast_2d(x_1)
        length_scale = self._check_length_scale(x_1, len_scale)
        if x_2 is None:
            dists = pdist(x_1 / length_scale, metric='sqeuclidean')
            k = sigma_f * np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            k = squareform(k)
            np.fill_diagonal(k, 1)
        else:
            dists = cdist(x_1 / length_scale, x_2 / length_scale, metric='sqeuclidean')
            k = sigma_f * np.exp(-.5 * dists)

        return k

    def _squared_exponential_kernel_cs(self, x_1, x_2):
        """
        Symbolic implementation of the anisotropic squared exponential kernel
        :param x_1: Array of m points (m x d).
        :param x_2: Array of n points (m x d).
        :return: Covariance matrix (m x n).
        """

        # Length scale parameter
        len_scale = self.params['l'] if 'l' in self.params.keys() else 1.0
        # Vertical variation parameter
        sigma_f = self.params['sigma_f'] if 'sigma_f' in self.params.keys() else 1.0

        if x_1.shape != x_2.shape and x_2.shape[0] == 1:
            tiling_ones = cs.MX.ones(x_1.shape[0], 1)
            d = x_1 - cs.mtimes(tiling_ones, x_2)
            dist = cs.sum2(d ** 2 / cs.mtimes(tiling_ones, cs.MX(len_scale ** 2).T))
        else:
            d = x_1 - x_2
            dist = cs.sum1(d ** 2 / cs.MX(len_scale ** 2))

        return sigma_f * cs.SX.exp(-.5 * dist)

    def diff(self, z, z_train):
        """
        Computes the symbolic differentiation of the kernel function, evaluated at point z and using the training
        dataset z_train. This function implements equation (80) from overleaf document, without the c^{v_x} vector,
        and for all the partial derivatives possible (m), instead of just one.

        :param z: evaluation point. Symbolic vector of length m
        :param z_train: training dataset. Symbolic matrix of shape n x m

        :return: an m x n matrix, which is the derivative of the exponential kernel function evaluated at point z
        against the training dataset.
        """

        if self.kernel_type != 'squared_exponential':
            raise NotImplementedError

        len_scale = self.params['l'] if len(self.params['l']) > 0 else self.params['l'] * cs.MX.ones(z_train.shape[1])
        len_scale = np.atleast_2d(len_scale ** 2)

        # Broadcast z vector to have the shape of z_train (tile z to to the number of training points n)
        z_tile = cs.mtimes(cs.MX.ones(z_train.shape[0], 1), z.T)

        # Compute k_zZ. Broadcast it to shape of z_tile and z_train, i.e. by the number of variables in z.
        k_zZ = cs.mtimes(cs.MX.ones(z_train.shape[1], 1), self.__call__(z_train, z.T).T)

        return - k_zZ * (z_tile - z_train).T / cs.mti
    
    
