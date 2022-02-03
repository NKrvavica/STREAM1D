# -*- coding: utf-8 -*-
"""

Created on Tue Feb 20 11:57:09 2018

@author: Nino Krvavica

MAIN STREAM-1D SCRIPT
"""

import sys
import warnings
import os
import timeit
import glob
import roe
import interfacial_friction
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.interpolate import interp1d
from scipy import optimize
from stream_geometry import get_geometry
from numba import jit

plt.close('all')



# --------------------------------------------------------------------------- #
# MATRIX RECOMPOSITION
# --------------------------------------------------------------------------- #
def recompose_matrix(K, L):
    '''
    Recomposese the matrix from known eigenvalues and eigevectors.
    It also composes matrices when eigenvalue matrix is modified.

    K: stacked array, columns are right eigenvectors
    L: stacked diagonal array, where diagonal values are eigenvalues

    Output is the same as K @ L @ inv(K)
    '''
    A = K.transpose(0, 2, 1)
    B = (L[:, None, :] * K).transpose(0, 2, 1)
    return np.linalg.solve(A, B).transpose(0, 2, 1)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# MATRIX MULTIPLICATION
# --------------------------------------------------------------------------- #
def M_x_v(M, v):
    '''
    Batch Matrix multiplication of matrix and vector
    M: stacked 2D array
    v: stacked vector
    '''
    return np.einsum("...ij, ...j -> ...i", M, v)
    # return np.einsum("...ij, j... -> ...i", M, v)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# HARTEN REGULARIZATION
# --------------------------------------------------------------------------- #
def harten_regularization(Lam, eps=1e-1):
    ''' Harten regularization is needed if one of the eigenvelues is zero'''
    A_abs = np.abs(Lam)
    A_abs += (0.5 * ((1 + np.sign(eps - A_abs))
              * ((Lam**2 + eps*eps) / (2*eps) - A_abs)))
    return A_abs
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# LOAD INPUT FILE
# --------------------------------------------------------------------------- #
def load_inputs(input_file):
    Par = namedtuple('Par', ['g', 'r', 'wfric', 'twolayers', 'ks', 'fi_eq',
                             'vis', 'Lstart', 'Lend', 'dx', 'T', 'CFL',
                             't_rec', 'solver', 'init'])
    inputi = []
    with open(input_file) as f:
        for line in f:
            podaci = line.split(' = ')
            inputi.append(float(podaci[-1]))
    return Par(*inputi)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# SET SPATIAL PARAMETERS
# --------------------------------------------------------------------------- #
def set_spatial_domain(p):
    Space = namedtuple('Space', ['L', 'ndx', 'x_uk', 'x'])
    # total length
    L = p.Lend - p.Lstart
    # number of spatial steps
    ndx = round(L / p.dx)
    # vector of total x coordinates (with ghost cells)
    x_uk = np.arange(p.Lstart, p.Lend + 2*p.dx, p.dx) - p.dx/2
    # x without boundary (ghost) cells
    x = x_uk[1:-1]
    return Space(L, ndx, x_uk, x)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# CONVERT FUNCTIONS
# --------------------------------------------------------------------------- #
# compute flow rate q from known velocity u
def u2q(u, a):
    q = u * a
    return q


# compute velocity u from known flow rate q
def q2u(q, a):
    if a.size == 1:
        if a > 0:
            u = q / a
        else:
            u = 0
    else:
        u = np.zeros_like(q)
        mask = a > 0
        u[mask] = q[mask] / a[mask]
    return u


@jit(nopython=True)
def interp_geom_multi(x, xi, yi):
    values = np.zeros(len(x))
    for n in range(len(x)):
        values[n] = interp_geom(x[n], xi, yi[:, n])
    return values


@jit(nopython=True)
def interp_geom(x, xi, yi):
    temp = (x - xi[0]) / (xi[-1] - xi[0]) * (len(xi) - 1)
    idx_down = int(np.floor(temp))
    y_down = yi[idx_down]
    y_up = yi[idx_down + 1]
    return y_down + (y_up - y_down) * (temp - idx_down)


def interp_geom_multi2(x, xi, yi):
    temp = (x - xi[0]) / (xi[-1] - xi[0]) * (len(xi) - 1)
    idx_down = np.floor(temp).astype('int')
    cols =  np.arange(len(x))
    y_down = yi[idx_down, cols]
    y_up = yi[idx_down + 1, cols]
    return y_down + (y_up - y_down) * (temp - idx_down)

def get_sigma(idx, surface, interface, geo_mat, r, g):
    sig1 = interp_geom(surface, geo_mat.yi, geo_mat.B[:, idx])
    sig3 = interp_geom(interface, geo_mat.yi, geo_mat.B[:, idx])
    sig2 = 1 / ((1 - r) / sig3 + r / sig1)
    return np.array([sig1, sig2, sig3])


def get_perim(idx, surface, interface, geo_mat, r, g):
    o_tot = interp_geom(surface, geo_mat.yi, geo_mat.O[:, idx])
    o2 = interp_geom(interface, geo_mat.yi, geo_mat.O[:, idx])
    o1 = o_tot - o2
    return np.array([o1, o2])
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# COMPUTE ALL HYDRAULIC PARAMETERS
# --------------------------------------------------------------------------- #
def compute_hyd_param(w, geo_mat, b, r):
    # correct negative depths
    negative_depths = w[:, 2] < 0
    w[negative_depths, 2] = 0
    w[negative_depths, 3] = 0

    # water levels
    surface = interp_geom_multi(w[:, 0] + w[:, 2], geo_mat.ai, geo_mat.D)
    interface = interp_geom_multi(w[:, 2], geo_mat.ai, geo_mat.D)

    # velocities
    u1 = q2u(w[:, 1], w[:, 0])
    u2 = q2u(w[:, 3], w[:, 2])

    # depths
    h1 = surface - interface
    h2 = interface - b
    w_ = np.column_stack((h1, u1, h2, u2))

    # channel width
    sig1 = interp_geom_multi(surface, geo_mat.yi, geo_mat.B)
    sig3 = interp_geom_multi(interface, geo_mat.yi, geo_mat.B)
    sig2 = 1 / ((1 - r) / sig3 + r / sig1)
    sig = np.vstack((sig1, sig2, sig3))

    # wetted perimiter
    o_tot = interp_geom_multi(surface, geo_mat.yi, geo_mat.O)
    o2 = interp_geom_multi(interface, geo_mat.yi, geo_mat.O)
    if (o2 == sig3).all():
        o1 = o_tot
    else:
        o1 = o_tot - o2
    o = np.vstack((o1, o2))

    return w_, sig, o
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# LOAD AND SET INITAL CONDITIONS
# --------------------------------------------------------------------------- #
def load_init_cond(x_uk, geo_mat, b, p, plot_initial=False):
    # depth and velocity
    w = np.loadtxt('initial.txt').T

    if p.r == 0:
        r_series = np.loadtxt(rfile)
        r = r_series[0, 1]
    else:
        r = p.r # take fixed value

    w_, sig, o = compute_hyd_param(w, geo_mat, b, r)
    # plot inital condition
    if plot_initial:
        plt.figure()
        plt.plot(x_uk, b)
        plt.plot(x_uk, b + w_[:, 2])
        plt.plot(x_uk, b + w_[:, 2] + w_[:, 0])
    return w, w_, sig, o
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# LOAD BOUNDARY CONDITIONS
# --------------------------------------------------------------------------- #
def load_boun_cond(p):
    BC_types = []
    with open('boundary.txt') as f:
        for line in f:
            podaci = line.split(' = ')
            BC_types.append(int(podaci[-1]))
    BC_files = ['BC_h1_start.txt', 'BC_h2_start.txt', 'BC_q1_start.txt',
                'BC_q2_start.txt', 'BC_h1_end.txt', 'BC_h2_end.txt',
                'BC_q1_end.txt', 'BC_q2_end.txt']

    # generate boundary conditions
    """
    BC MATRIX :
        1. row - type of BC (0 - zero flux gradient ,
                             1 - time series from a file,
                             4 - critical condition)
        1. column - time vector
        2. column [0] - BC values for upstream depth h1
        3. column [1] - BC values for downstream depth h1
        4. column [2] - BC values for upstream flow rate q1
        5. column [3] - BC values for downstream flow rate q1
        6. column [4] - BC values for upstream depth h2 (or H)
        7. column [5] - BC values for downstream depth h2 (or H)
        8. column [6] - BC values for upstream flow rate q2
        9. column [7] - BC values for downstream flow rate q2
    """

    BC = []
    for i, BC_file in enumerate(BC_files):
        # if BC is given in a file
        if BC_types[i] == 1 or BC_types[i] == 11:
            BC_time_series = np.loadtxt(BC_file)
            BC.append(interp1d(BC_time_series[:, 0], BC_time_series[:, 1]))

    if p.r == 0:
        r_series = np.loadtxt(rfile)
    else:
        r_series = []

    return np.array(BC_types), BC, r_series
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# INITIALIZE MEMORY
# --------------------------------------------------------------------------- #
def initialize_memory(w, w_, t):
    W = [w]
    W_ = [w_]
    TIME = [t]
    return W, W_, TIME
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# WRITE RESULTS
# --------------------------------------------------------------------------- #
def write_results(W, W_, w, w_, TIME, t):
    W = np.append(W, [w], axis=0)
    W_ = np.append(W_, [w_], axis=0)
    TIME = np.append(TIME, t)
    return W, W_, TIME
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FIND DRY AND WET CELLS
# --------------------------------------------------------------------------- #
def correct_negative_depths(w, tol=1e-6, minB=1, min_dubina=1e-3, entrain=True):
    # correct negative depths
    negative_depths1 = w[:, 0] < 0
    w[negative_depths1, :2] = 0
    negative_depths2 = w[:, 2] < 0
    w[negative_depths2, 2:] = 0

    # correct very small depths
    negative_depths2 = w[:, 2] < tol
    w[negative_depths2, 3] = 0
    w[negative_depths2, 0] = w[negative_depths2, 0] + w[negative_depths2, 2]
    w[negative_depths2, 2] = 0
    negative_depths1 = w[:, 0] < tol
    w[negative_depths1, :2] = 0

    # entraine lower layer depths
    if entrain:
        thin_lower = w[:, 2] < minB * min_dubina
        same_direction = np.sign(w[:, 1]) == np.sign(w[:, 3])
        to_entrain = thin_lower & same_direction
        w[to_entrain, 0] += w[to_entrain, 2]
        w[to_entrain, 2] = 0
        w[to_entrain, 3] = 0

    return w


def find_dry_cells(w, w_, b, min_dubina):

    # find wet cells
    wet_cells1 = (w_[:, 0] > min_dubina)    # h_1
    wet_cells2 = (w_[:, 2] > min_dubina)    # h_2

    # corect the flow rates and velocities in dry cells
    w_[~wet_cells1, 1] = 0                  # u_1
    w[~wet_cells1, 1] = 0                   # Q_1
    w_[~wet_cells2, 3] = 0                  # u_2
    w[~wet_cells2, 3] = 0                   # Q_2

    '''
    Possible types of flow structure and its ID:
    0 - all dry
    1 - upper layer wet-dry
    10 - lower layer wet-dry
    2 - upper layer wet-wet
    20 - lower layer wet-wet
    11 - two-layer wet-dry (both layers dry)
    12 - two-layer wet-dry (lower layer dry)
    21 - two-layer wet-dry( upper layer dry) , treat as wet-wet
    22 - two-layer wet-wet
    '''
    temp = wet_cells1.astype(int) + wet_cells2.astype(int)*10
    interface_case = temp[:-1] + temp[1:]

    return w, w_, wet_cells1, wet_cells2, interface_case
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# COMPUTE UNKNOWNS
# --------------------------------------------------------------------------- #
def compute_fluxes(ndx, w, sig, wet_cells1, wet_cells2, g):
    # initialize
    Fn = np.zeros((ndx+2, 4))
    # numerical flux
    Fn[:, 0] = w[:, 1]
    Fn[:, 1] = 0.5 * g / sig[0, :] * w[:, 0]**2
    Fn[:, 2] = w[:, 3]
    Fn[:, 3] = 0.5 * g / sig[1, :] * w[:, 2]**2

    Fn[wet_cells1, 1] = (w[wet_cells1, 1]**2 / w[wet_cells1, 0]
                         + Fn[wet_cells1, 1])
    Fn[wet_cells2, 3] = (w[wet_cells2, 3]**2 / w[wet_cells2, 2]
                         + Fn[wet_cells2, 3])
    return Fn
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# ONE LAYER CASE
# --------------------------------------------------------------------------- #
def find_eig(u1, c1_2, br_lin):
    '''
    Compute eigenvalues and eigenvactors for 2x2 matrix defined by
    u1 nad h1
    '''
    # matrix of eigenvalues at diagonals
    L = np.zeros([br_lin, 2, 2])
    c = np.sqrt(c1_2)
    L[:, 0, 0] = u1 - c
    L[:, 1, 1] = u1 + c
    # matrix of eigenvectors
    K = np.zeros([br_lin, 2, 2])
    K[:, 0, :] = 1
    K[:, 1, 0] = u1 - c
    K[:, 1, 1] = u1 + c
    # inverse of eigenvactor matrix
    K_inv = np.zeros([br_lin, 2, 2])
    K_inv[:, 0, 0] = (u1 + c) / (2*c)
    K_inv[:, 0, 1] = -0.5 / c
    K_inv[:, 1, 0] = (- u1 + c) / (2*c)
    K_inv[:, 1, 1] = 0.5 / c
    return L, K, K_inv


def get_aRoe_one(br_lin, u1pol, c1pol_2):
    Id = np.eye(2, 2)

    # analytic eigen
    L_mat, K, K_inv = find_eig(u1pol, c1pol_2, br_lin)
    Lam = np.zeros([br_lin, 2])
    Lam[:, 0] = L_mat[:, 0, 0]
    Lam[:, 1] = L_mat[:, 1, 1]

    A_abs = harten_regularization(Lam)

    # Recompose
    sign_A = np.sign(Lam)
    Q = recompose_matrix(K, A_abs)
    C = recompose_matrix(K, sign_A)
    Pp = 0.5 * (Id + C)  # positive part of the projection matrix
    Pm = 0.5 * (Id - C)  # negative part of the projection matrix

    return Q, Pp, Pm, Lam


def linearize_one_layer(w, w_, f, sig, o, b, p, idx, modification):

    A1, Q1, A2, Q2 = w.T
    h1, u1, h2, u2 = w_.T
    sig1, sig2, sig3 = sig
    o1, o2 = o

    cells_L = np.array(idx)
    cells_R = cells_L + 1
    if modification == 1:
        to_correct_L = np.where(h1[cells_R] < b[cells_L] - b[cells_R])
        to_correct_R = np.where(h1[cells_L] < b[cells_R] - b[cells_L])
    else:
        to_correct_L = []
        to_correct_R = []

    A1pol = 0.5 * (A1[cells_L] + A1[cells_R])
    h1pol = 0.5 * (h1[cells_L] + h1[cells_R])
    sqrt_aL = np.sqrt(A1[cells_L])
    sqrt_aR = np.sqrt(A1[cells_R])
    u1pol = (sqrt_aL*u1[cells_L] + sqrt_aR*u1[cells_R]) / (sqrt_aL + sqrt_aR)
    if to_correct_L:
        w[cells_R[to_correct_L], 1] = 0
        u1pol[to_correct_L] = 0
    if to_correct_R:
        w[cells_L[to_correct_R], 1] = 0
        u1pol[to_correct_R] = 0

    sig1pol = 0.5 * (sig1[cells_L] + sig1[cells_R])
    c1pol_2 = p.g * A1pol / sig1pol

    br_lin = len(h1pol)
    Vpol = np.zeros([br_lin, 2])
    Vpol[:, 1] = 0.5 * p.g * ((1/sig1[cells_R] - 1/sig1pol) * A1[cells_R]**2
                              + (1/sig1pol - 1/sig1[cells_L]) * A1[cells_L]**2)

    Q, Pp, Pm, Lam = get_aRoe_one(br_lin, u1pol, c1pol_2)

    # numerical flux
    Wn_diff = (w[cells_R, :2] - w[cells_L, :2])
    Fpol_flux = (f[cells_R, :2] + f[cells_L, :2])
    Fpol_visc = M_x_v(Q, Wn_diff)
    Fpol = 0.5 * (Fpol_flux - Fpol_visc)

    # geometry source term
    Spol = np.zeros((br_lin, 2))
    Spol[:, 1] = p.g * A1pol * (((A1[cells_R] - A1[cells_L])
                                + (A2[cells_R] - A2[cells_L])) / sig1pol
                                - (b[cells_R] - b[cells_L]
                                + h1[cells_R] - h1[cells_L]
                                + h2[cells_R] - h2[cells_L]))

    if to_correct_L:
        Spol[to_correct_L, 1] = (p.g*A1pol[to_correct_L]
                                 * 1/sig1pol[to_correct_L]
                                 * A1[cells_R[to_correct_L]])
    if to_correct_R:
        Spol[to_correct_R, 1] = (- p.g*A1pol[to_correct_R]
                                 * 1/sig1pol[to_correct_R]
                                 * A1[cells_L[to_correct_R]])
    # hydraulic parameters
    O1pol = 0.5 * (o1[cells_L] + o1[cells_R])
    R1pol = A1pol / O1pol

    # friction source term
    Sfpol = np.zeros([br_lin, 2])
    if p.wfric == 1:  # Manning
        Sfpol[:, 1] -= (p.g * A1pol * p.ks**2 / R1pol**(4/3)
                        * (u1pol) * np.abs(u1pol))
    elif p.wfric == 2:  # Darcy-Weisbach
        Re1pol = np.abs(u1pol) * R1pol / p.vis
        fb = 1/32 * (-np.log10(p.ks / (12 * R1pol) + 1.95 / Re1pol**0.9))**(-2)
        Sfpol[:, 1] -= fb * (u1pol) * np.abs(u1pol) * O1pol
    elif p.wfric == 0:
        Sfpol[:, 1] = Sfpol[:, 1]
    else:
        print('wrong type of friction (1 or 2)')

    return Fpol, Vpol, Pp, Pm, Spol, Sfpol, Lam
# --------------------------------------------------------------------------- #


# Roe's linearizitation for two layers
def linearize_two(w, w_, f, sig, o, b, p, r, idx, Id4, get_solv, scheme,
                  eig_type, hyp_corr, modification):

    A1, Q1, A2, Q2 = w.T
    h1, u1, h2, u2 = w_.T
    sig1, sig2, sig3 = sig
    o1, o2 = o

    cells_L = np.array(idx)
    cells_R = cells_L + 1

    H = h1 + h2
    if modification == 1:
        to_correct1_L = np.where(h2[cells_R] < b[cells_L] - b[cells_R])
        to_correct1_R = np.where(h2[cells_L] < b[cells_R] - b[cells_L])
        to_correct2_L = []
        to_correct2_R = []
    elif modification == 2:
        to_correct1_L = np.where((h2[cells_R] < b[cells_L] - b[cells_R])
                                 & (H[cells_R] > b[cells_L] - b[cells_R]))
        to_correct1_R = np.where((h2[cells_L] < b[cells_R] - b[cells_L])
                                 & (H[cells_L] > b[cells_R] - b[cells_L]))
        to_correct2_L = np.where(H[cells_R] < b[cells_L] - b[cells_R])
        to_correct2_R = np.where(H[cells_L] < b[cells_R] - b[cells_L])
    else:
        to_correct1_L = []
        to_correct1_R = []
        to_correct2_L = []
        to_correct2_R = []

    A1pol = 0.5 * (A1[cells_L] + A1[cells_R])
    A2pol = 0.5 * (A2[cells_L] + A2[cells_R])

    h1pol = 0.5 * (h1[cells_L] + h1[cells_R])
    h2pol = 0.5 * (h2[cells_L] + h2[cells_R])

    sqrt_a1L = np.sqrt(A1[cells_L])
    sqrt_a1R = np.sqrt(A1[cells_R])
    sqrt_a2L = np.sqrt(A2[cells_L])
    sqrt_a2R = np.sqrt(A2[cells_R])

    u1pol = ((sqrt_a1L * u1[cells_L] + sqrt_a1R * u1[cells_R])
             / (sqrt_a1L + sqrt_a1R))
    u2pol = ((sqrt_a2L * u2[cells_L] + sqrt_a2R * u2[cells_R])
             / (sqrt_a2L + sqrt_a2R))
    Ri = p.g * (1 - r) * h1pol / (u1pol-u2pol + 1e-6)**2

    if to_correct1_L:
        w[cells_R[to_correct1_L], 3] = 0
        u2pol[to_correct1_L] = 0
    if to_correct1_R:
        w[cells_L[to_correct1_R], 3] = 0
        u2pol[to_correct1_R] = 0
    if to_correct2_L:
        w[cells_R[to_correct2_L], 1] = 0
        w[cells_R[to_correct2_L], 3] = 0
        u1pol[to_correct2_L] = 0
        u2pol[to_correct2_L] = 0
        Ri[to_correct2_L] = 0
    if to_correct2_R:
        w[cells_L[to_correct2_R], 1] = 0
        w[cells_L[to_correct2_R], 3] = 0
        u1pol[to_correct2_R] = 0
        u2pol[to_correct2_R] = 0
        Ri[to_correct2_R] = 0

    sig1pol = 0.5 * (sig1[cells_L] + sig1[cells_R])
    sig2pol = 0.5 * (sig2[cells_L] + sig2[cells_R])
    sig3pol = 0.5 * (sig3[cells_L] + sig3[cells_R])

    c1pol_2 = p.g / sig1pol * A1pol
    c2pol_2 = p.g / sig1pol * A2pol
    c22pol_2 = p.g / sig2pol * A2pol

    br_lin = len(h1pol)

    # matrix B
    Bpol = np.zeros([br_lin, 4, 4])
    Bpol[:, 1, 2] = - c1pol_2
    Bpol[:, 3, 0] = - r * c2pol_2

    if scheme == 'roe' and eig_type != 'numerical':
        A = np.zeros([br_lin, 4, 4])
    else:
        ''' ovaj dio ne treba za ARoe'''
        # Jasobian matrix
        Jpol = np.zeros([br_lin, 4, 4])
        Jpol[:, 0, 1] = 1
        Jpol[:, 1, 0] = - u1pol**2 + c1pol_2
        Jpol[:, 1, 1] = 2 * u1pol
        Jpol[:, 2, 3] = 1
        Jpol[:, 3, 2] = - u2pol**2 + c22pol_2
        Jpol[:, 3, 3] = 2 * u2pol

        # numerical flux matrix
        A = Jpol - Bpol
        ''' do tu...'''

    (Q, Pp, Pm, Lam, Fmod) = get_solv.comp_Q(u1pol, u2pol, c1pol_2, c2pol_2,
                                             c22pol_2, r, p.g, A,
                                             eig_type=eig_type, scheme=scheme,
                                             hyp_corr=hyp_corr)

    # matrix V
    Vpol = np.zeros([br_lin, 4])
    Vpol[:, 1] = 0.5 * p.g * ((1/sig1[cells_R] - 1/sig1pol) * A1[cells_R]**2
                              + (1/sig1pol - 1/sig1[cells_L]) * A1[cells_L]**2)
    Vpol[:, 3] = 0.5 * p.g * ((1/sig2[cells_R] - 1/sig2pol) * A2[cells_R]**2
                              + (1/sig2pol - 1/sig2[cells_L]) * A2[cells_L]**2)

    # numerical flux
    Wn_diff = (w[cells_R, :] - w[cells_L, :])
    Fpol_flux = (f[cells_R, :] + f[cells_L, :])
    if get_solv == roe:
        ''' za Roe method'''
        Fpol_visc = M_x_v(Q, Wn_diff)
    Fpol = 0.5 * (Fpol_flux - Fpol_visc)

    # bed elevation step
    delta_b = b[cells_R] - b[cells_L]
    delta_b1 = delta_b.copy()
    delta_b2 = delta_b.copy()

    # wet-dry source correction
    if to_correct1_R:
        delta_b2[to_correct1_R] = (r * delta_b[to_correct1_R]
                                   + (1-r) * h2[cells_L[to_correct1_R]])
    if to_correct1_L:
        delta_b2[to_correct1_L] = (r * delta_b[to_correct1_L]
                                   - (1-r) * h2[cells_R[to_correct1_L]])
    if to_correct2_R:
        delta_b1[to_correct2_R] = (h1[cells_L[to_correct2_R]]
                                   + h2[cells_L[to_correct2_R]])
        delta_b2[to_correct2_R] = (r * h1[cells_L[to_correct2_R]]
                                   + h2[cells_L[to_correct2_R]])
    if to_correct2_L:
        delta_b1[to_correct2_L] = - (h1[cells_R[to_correct2_L]]
                                     + h2[cells_R[to_correct2_L]])
        delta_b2[to_correct2_L] = - (r * h1[cells_R[to_correct2_L]]
                                     + h2[cells_R[to_correct2_L]])

    # geometry source term
    Spol = np.zeros([br_lin, 4])
    Spol[:, 1] = p.g * A1pol * ((Wn_diff[:, 0] + Wn_diff[:, 2]) / sig1pol
                                - (delta_b1 + h1[cells_R] - h1[cells_L]
                                   + h2[cells_R] - h2[cells_L]))
    Spol[:, 3] = p.g * A2pol * (Wn_diff[:, 2] / sig2pol
                                + r * (Wn_diff[:, 0]) / sig1pol
                                - (delta_b2 + r * (h1[cells_R] - h1[cells_L])
                                   + h2[cells_R] - h2[cells_L]))

    # initialise source term for friction and entrainment
    Sfpol = np.zeros([br_lin, 4])

    # interfacial friction
    Q1pol = ((sqrt_a1L * Q1[cells_L] + sqrt_a1R * Q1[cells_R])
             / (sqrt_a1L + sqrt_a1R))
    Q2pol = ((sqrt_a2L * Q2[cells_L] + sqrt_a2R * Q2[cells_R])
             / (sqrt_a2L + sqrt_a2R))

    fi = interfacial_friction.compute_fi(p.fi_eq, Q1pol, Q2pol, A1pol, A2pol,
                                          u1pol, u2pol, h1pol, h2pol, r)

    # Friction
    koef = 1
    delta_u = u1pol - u2pol
    Sfpol[:, 1] -= (fi * koef * delta_u
                    * np.abs(delta_u) * sig3pol)
    Sfpol[:, 3] += (fi * r * koef * delta_u
                    * np.abs(delta_u) * sig3pol)

    if to_correct1_L:
        Sfpol[to_correct1_L, 1] = 0
        Sfpol[to_correct1_L, 3] = 0
    if to_correct1_R:
        Sfpol[to_correct1_R, 1] = 0
        Sfpol[to_correct1_R, 3] = 0
    if to_correct2_L:
        Sfpol[to_correct2_L, 1] = 0
        Sfpol[to_correct2_L, 3] = 0
    if to_correct2_R:
        Sfpol[to_correct2_R, 1] = 0
        Sfpol[to_correct2_R, 3] = 0

    # hydraulic parameters
    O1pol = 0.5 * (o1[cells_L] + o1[cells_R])
    R1pol = A1pol / O1pol
    O2pol = 0.5 * (o2[cells_L] + o2[cells_R])
    R2pol = A2pol / O2pol
    zero_mask = (A2pol == 0)
    R2pol[zero_mask] = sig1pol[zero_mask]

    # bed friction source term
    if p.wfric == 1:  # Manning
        Sfpol[:, 1] -= (p.g * A1pol * p.ks**2 / R1pol**(4/3)
                        * (u1pol) * np.abs(u1pol))
        Sfpol[:, 3] -= (p.g * A2pol * p.ks**2 / R2pol**(4/3)
                        * (u2pol) * np.abs(u2pol))
    elif p.wfric == 2:  # Darcy-Weisbach
        Re1pol = np.abs(u1pol) * h1pol / p.vis
        fb1 = 1/32 * (-np.log10(p.ks / (12 * R1pol) + 1.95 / Re1pol**0.9))**(-2)
        Sfpol[:, 3] -= fb1 * (u1pol) * np.abs(u1pol) * O1pol
        Re2pol = np.abs(u2pol) * R2pol / p.vis
        fb2 = 1/32 * (-np.log10(p.ks / (12 * R2pol) + 1.95 / Re2pol**0.9))**(-2)
        Sfpol[:, 3] -= fb2 * (u2pol) * np.abs(u2pol) * O2pol
    elif p.wfric == 0:
        pass
    else:
        print('krivo zadan tip trenja (1 ili 2)')

    Fcorr = np.zeros([br_lin, 4])
    Fcorr[:, 1] = - Fmod[:, 0] * np.sign(u1pol - u2pol) * sig3pol
    Fcorr[:, 3] = r * Fmod[:, 0] * np.sign(u1pol - u2pol) * sig3pol

    return Fpol, Bpol, Vpol, Pp, Pm, Spol, Sfpol, Lam, Fcorr
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# CHECK CFL CONDITION
# --------------------------------------------------------------------------- #
def compute_time_step(p, Lam):
    return p.CFL * p.dx / np.abs(Lam).max()
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# ROE's SOLVER
# --------------------------------------------------------------------------- #
def Roe_solver(p, dt, space, w, Fpol, Bpol, Vpol, Pp, Pm, Spol, Sfpol):
    dt_dx = dt / p.dx
    Wn1 = np.zeros((space.ndx + 2, 4))
    Wn1[1:-1, :] = (
                    w[1:-1, :]
                    + dt_dx * (Fpol[:-1, :] - Fpol[1:, :])
                    + 0.5 * dt_dx * (M_x_v(Bpol[:-1, :, :],
                                           w[1:-1, :] - w[:-2, :])
                                     + M_x_v(Bpol[1:, :, :],
                                             w[2:, :] - w[1:-1, :]))
                    + 0.5 * dt_dx * (Vpol[:-1, :] + Vpol[1:, :])
                    + dt_dx * (M_x_v(Pp[:-1, :, :], Spol[:-1, :])
                               + M_x_v(Pm[1:, :, :], Spol[1:, :]))
                    + dt * (M_x_v(Pp[:-1, :, :], Sfpol[:-1, :])
                            + M_x_v(Pm[1:, :, :], Sfpol[1:, :]))
                    )
    return Wn1
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# HYPERBOLICITY CORRECTION
# --------------------------------------------------------------------------- #
def Corr_ARoe(Wn1, sig1, Pp, Pm, Fcorr):
    Wn1[1:-1, :] = (Wn1[1:-1, :]
                    + (M_x_v(Pp[:-1, :, :], Fcorr[:-1, :])
                        + M_x_v(Pm[1:, :, :], Fcorr[1:, :])))
    print('.', end='')
    return Wn1
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FUNCTION: SET BOUNDARY CONDITIONS
# --------------------------------------------------------------------------- #

def f_critical(interface, surface, Q1, Q2, A_tot, sig1, geo_mat, r, g, idx):
    A2 = interp_geom(interface, geo_mat.yi, geo_mat.A[:, idx])
    A1 = A_tot - A2
    sig3 = interp_geom(interface, geo_mat.yi, geo_mat.B[:, idx])
    sig2 = 1 / ((1-r) / sig3 + r / sig1)
    Fd1 = np.abs(Q1)*Q1 * sig1 * sig3 / (g * (1 - r) * A1**3 * sig2)
    if A2 == 0:
        Fd2 = 0
    else:
        Fd2 = np.abs(Q2)*Q2 * sig3 / (g * (1 - r) * A2**3)
    return np.abs(Fd1 - Fd2) - 1

def newton_secant(func, x0, fprime=None, args=(), step=1e-4, tol=1.48e-8,
                  maxiter=50, fprime2=None):
    # Secant method
    p0 = 1.0 * x0
    if x0 >= 0:
        p1 = x0*(1 + step) + step
    else:
        p1 = x0*(1 + step) - step
    q0 = func(p0, *args)
    q1 = func(p1, *args)
    for iter in range(maxiter):
        if q1 == q0:
            if p1 != p0:
                msg = "Tolerance of %s reached" % (p1 - p0)
                warnings.warn(msg, RuntimeWarning)
            return (p1 + p0)/2.0
        else:
            p = p1 - q1*(p1 - p0)/(q1 - q0)
        if abs(p - p1) < tol:
            return p
        p0 = p1
        q0 = q1
        p1 = p
        q1 = func(p1, *args)
    msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
    warnings.warn(msg, RuntimeWarning)
    return p


def find_H_critical(surface, Q1, Q2, sig, bottom, geo_mat, r, g, idx=0):
    # first guess
    A_tot = interp_geom(surface, geo_mat.yi, geo_mat.A[:, idx])
    sig2 = 1 / ((1-r) / minB + r / sig[0])
    Fd1 = Q1**2 * sig[0] * minB / (g * (1 - r) * A_tot**3 * sig2)
    if Fd1 >= 0.95:
        return bottom
    else:
        sig_mean = np.mean(sig)
        A1_0 = (Q1**2 * sig_mean / (g * (1-r)))**(1/3)
        A2_0 = max(0, A_tot - A1_0)
        a = interp_geom(A2_0, geo_mat.ai, geo_mat.D[:, idx])
        return newton_secant(f_critical, a,
                             args=(surface, Q1, Q2, A_tot, sig[0], geo_mat,
                                   r, g, idx),
                             step=1e-9, tol=1e-3)


def set_BC(BC_types, BC, t, dt, w, w_, wn, sig, o, geo_mat, p, r,
           two_layers=True):

    if (BC_types == 0).all():
        ''' Free boundary conditions:
        Zero gradient flux'''
        w[0, :] = w[1, :]
        w_[0, :] = w_[1, :]
        sig[:, 0] = sig[:, 1]
        o[:, 0] = o[:, 1]

        w[-1, :] = w[-2, :]
        w_[-1, :] = w_[-2, :]
        sig[:, -1] = sig[:, -2]
        o[:, -1] = o[:, -2]

    else:
        ''' Estuarine boundary conditions:
        Downstream:
             Total depth from timeseries
             Interface - internally critical depth
             Flow rates - zero gradient flux
        Upstream:
            Only one layer (upper)
            Flow rate from time series
            Total depth - zero gradient flux '''

        # downstream boundary conditions
        bottom = b[0]
        surface = BC[0](t)
        w[0, 1] = w[1, 1]                               # Q_1
        w[0, 3] = w[1, 3]                               # Q_2

        if two_layers==True:
            interface = find_H_critical(surface, w[0, 1],
                                        w[0, 3], sig[:, 1],
                                        bottom, geo_mat, r, p.g) - 5e-2
            interface = min(interface, surface-1e-1) # min upper layer depth = 0.1 m
            interface = max(interface, bottom)
        else:
            interface = bottom # this keeps only 1 layer (TURN OFF TWO-LAYER)

        w[0, 2] = interp_geom(interface, geo_mat.yi,
                              geo_mat.A[:, 0])            # A_2
        w[0, 0] = interp_geom(surface, geo_mat.yi,
                              geo_mat.A[:, 0]) - w[0, 2]    # A_1
        w_[0, 0] = surface - interface                  # h_1
        w_[0, 1] = q2u(w[0, 1], w[0, 0])                # u_1
        w_[0, 2] = interface - bottom                   # h_2
        w_[0, 3] = q2u(w[0, 3], w[0, 2])                # u_2
        sig[:, 0] = get_sigma(0, surface, interface, geo_mat, r, p.g)
        o[:, 0] = get_perim(0, surface, interface, geo_mat, r, p.g)

        # upstream boundary conditions
        inflow = BC[1](t)
        w[-1, 1] = inflow                                               # Q_1
        w[-1, 0] = wn[-1, 0] - dt/p.dx * (w[-1, 1] - w[-2, 1])        # A_1
        surface = interp_geom(w[-1, 0], geo_mat.ai, geo_mat.D[:, -1])   # h_1
        bottom = b[-1]
        w_[-1, 0] = surface - bottom
        w_[-1, 1] = q2u(w[-1, 1], w[-1, 0])                             # u_1
        interface = bottom
        sig[:, -1] = get_sigma(-1, surface, interface, geo_mat, r, p.g)
        o[:, -1] = get_perim(-1, surface, interface, geo_mat, r, p.g)

    return w, w_, sig, o
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# CHECK FOR STATIONARY CONDITIONS
# --------------------------------------------------------------------------- #
def check_if_stationary(q1, q1_, stacionarno):
    if np.mean(q1) != 0:
        if (np.max(np.abs(q1[1:-1] - q1_[1:-1])) > 1e-6
            or (np.max(np.abs(q1[1:-1])) - np.min(np.abs(q1[1:-1]))) > 1e-6):
            if stacionarno:
                stacionarno = 0
        else:
            stacionarno += 1
    return stacionarno
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# TIME STEPPING PROCEDURE
# --------------------------------------------------------------------------- #
# @profile
def time_stepping(p, r_series, space, geo_mat, b, w, w_, sig, o, BC_types, BC,
                  min_dubina=5e-3, two_layers=True):

    W, W_, TIME = initialize_memory(w, w_, 0)

    # some constants
    Id4 = np.dstack([np.eye(4, 4)] * (space.ndx + 1)).T

    if p.solver == 11:
        print('using ROE solver and numerical eigenvalues')
        get_solv = roe
        scheme = 'roe'
        eig_type = 'numerical'
    elif p.solver == 12:
        print('using ROE solver and analytical eigenvalues')
        get_solv = roe
        scheme = 'roe'
        eig_type = 'analytical'
    elif p.solver == 13:
        print('using ROE solver and approximated eigenvalues')
        get_solv = roe
        scheme = 'roe'
        eig_type = 'approximated'
    else:
        raise ValueError('bad solver code')

    t_Vector, F_max = [], []
    Tc, dtbr, t, trecbr, stacionarno = 0, 0, 1, 1, 0
    hyp_corr = True
    kraj = False
    start = timeit.default_timer()

    # if r=0 then a file is provided with a time series of r
    if p.r == 0:
        rt = interp1d(r_series[:, 0], r_series[:, 1])

    while Tc < p.T and not kraj:

        if p.r == 0:
            r = rt(Tc) # interpolate from a given time series
        else:
            r = p.r # take fixed value

        (w, w_, wet_cells1, wet_cells2,
         interface_case) = find_dry_cells(w, w_, b, min_dubina)

        f = compute_fluxes(space.ndx, w, sig, wet_cells1, wet_cells2, p.g)

        Fpol = np.zeros([space.ndx + 1, 4])
        Bpol = np.zeros([space.ndx + 1, 4, 4])
        Vpol = np.zeros([space.ndx + 1, 4])
        Pp = np.zeros([space.ndx + 1, 4, 4])
        Pm = np.zeros([space.ndx + 1, 4, 4])
        Spol = np.zeros([space.ndx + 1, 4])
        Sfpol = np.zeros([space.ndx + 1, 4])
        Lam = np.zeros([space.ndx + 1, 4])
        Fcorr = np.zeros([space.ndx + 1, 4])

        # wet-dry interfaces (L for dry-wet, R for wet-dry)
        upper_wd = np.where(interface_case == 1)[0]
        lower_wd = np.where(interface_case == 10)[0]
        upper = np.where(interface_case == 2)[0]
        lower = np.where(interface_case == 20)[0]
        two_both_wd = np.where(interface_case == 11)[0]
        two_lower_wd = np.where(interface_case == 12)[0]
        two_upper_wd = np.where(interface_case == 21)[0]
        two = np.where(interface_case == 22)[0]
        '''
        0 - all dry
        1 - upper layer wet-dry
        10 - lower layer wet-dry
        2 - upper layer wet
        20 - lower layer wet
        11 - two-layer wet-dry (both layers dry)
        12 - two-layer wet-dry (lower layer dry)
        21 - two-layer wet-dry( upper layer dry) , treat as wet-wet
        22 - two-layer wet
        '''

        if upper.size > 0:
            (Fpol[upper, :2], Vpol[upper, :2],
             Pp[upper, :2, :2], Pm[upper, :2, :2],
             Spol[upper, :2], Sfpol[upper, :2],
             Lam[upper, :2]) = linearize_one_layer(w, w_, f, sig, o, b, p,
                                                   upper, 0)

        if upper_wd.size > 0:
            (Fpol[upper_wd, :2], Vpol[upper_wd, :2],
             Pp[upper_wd, :2, :2], Pm[upper_wd, :2, :2],
             Spol[upper_wd, :2], Sfpol[upper_wd, :2],
             Lam[upper_wd, :2]) = linearize_one_layer(w, w_, f, sig, o, b, p,
                                                      upper_wd, 1)

        if lower.size > 0:
            (Fpol[lower, :2], Vpol[lower, :2],
             Pp[lower, :2, :2], Pm[lower, :2, :2],
             Spol[lower, :2], Sfpol[lower, :2],
             Lam[lower, :2]) = linearize_one_layer(w, w_, f, sig, o, b, p,
                                                   lower, 0)

        if lower_wd.size > 0:
            (Fpol[lower_wd, :2], Vpol[lower_wd, :2],
             Pp[lower_wd, :2, :2], Pm[lower_wd, :2, :2],
             Spol[lower_wd, :2], Sfpol[lower_wd, :2],
             Lam[lower_wd, :2]) = linearize_one_layer(w, w_, f, sig, o, b, p,
                                                      lower_wd, 1)

        if two.size > 0:
            (Fpol[two, :], Bpol[two, :, :], Vpol[two, :], Pp[two, :, :],
             Pm[two, :, :], Spol[two, :], Sfpol[two, :], Lam[two, :],
             Fcorr[two, :]) = linearize_two(w, w_, f, sig, o, b, p, r, two, Id4,
                                            get_solv, scheme, eig_type,
                                            hyp_corr, 0)

        if two_upper_wd.size > 0:
            (Fpol[two_upper_wd, :], Bpol[two_upper_wd, :, :],
             Vpol[two_upper_wd, :], Pp[two_upper_wd, :, :],
             Pm[two_upper_wd, :, :], Spol[two_upper_wd, :],
             Sfpol[two_upper_wd, :], Lam[two_upper_wd, :],
             Fcorr[two_upper_wd, :]) = linearize_two(w, w_, f, sig, o, b, p, r,
                                                     two_upper_wd, Id4,
                                                     get_solv, scheme,
                                                     eig_type, hyp_corr, 0)

        if two_lower_wd.size > 0:
            (Fpol[two_lower_wd, :], Bpol[two_lower_wd, :, :],
             Vpol[two_lower_wd, :], Pp[two_lower_wd, :, :],
             Pm[two_lower_wd, :, :], Spol[two_lower_wd, :],
             Sfpol[two_lower_wd, :], Lam[two_lower_wd, :],
             Fcorr[two_lower_wd, :]) = linearize_two(w, w_, f, sig, o, b, p, r,
                                                     two_lower_wd, Id4,
                                                     get_solv, scheme,
                                                     eig_type, hyp_corr, 1)

        if two_both_wd.size > 0:
            (Fpol[two_both_wd, :], Bpol[two_both_wd, :, :],
             Vpol[two_both_wd, :], Pp[two_both_wd, :, :],
             Pm[two_both_wd, :, :], Spol[two_both_wd, :],
             Sfpol[two_both_wd, :], Lam[two_both_wd, :],
             Fcorr[two_both_wd, :]) = linearize_two(w, w_, f, sig, o, b, p, r,
                                                    two_both_wd, Id4,
                                                    get_solv, scheme,
                                                    eig_type, hyp_corr, 2)

        # CFL CONDITION CONTROL
        dt = compute_time_step(p, Lam)
        Tc += dt
        dtbr += dt

        # ROE'S SOLVER FOR REIMANN
        wn = w.copy()
        w = Roe_solver(p, dt, space, w, Fpol, Bpol, Vpol, Pp, Pm, Spol, Sfpol)
        if (Fcorr != 0).any():
            w = Corr_ARoe(w, sig[0, :], Pp, Pm, Fcorr)

        # CORRECT NEGATIVE DEPTHS
        w = correct_negative_depths(w, minB=minB, min_dubina=min_dubina)

        # UPDATE CURRENTE TIME-STEP RESULTS (WITHOUT GHOST CELLS)
        w_, sig, o = compute_hyd_param(w, geo_mat, b, r)

        # ADD BOUNDARY CONDITIONS
        w, w_, sig, o = set_BC(BC_types, BC, Tc, dt, w, w_, wn,
                               sig, o, geo_mat, p, r, two_layers=two_layers)

        # SAVE RESULTS aND CHECK IF STATIONARY
        if dtbr >= p.t_rec:
            W, W_, TIME = write_results(W, W_, w, w_, TIME, Tc)

            stop = timeit.default_timer()
            remaining_t = (stop - start) / (Tc / p.T) - (stop - start)
            current_time = datetime.datetime.now()
            end_time = (current_time + datetime.timedelta(seconds=remaining_t))
            end_time = end_time.replace(microsecond=0)
            print(f't = {Tc:.3f} s / {p.T:.1f} s, (estimated end of simulation {end_time})')

            # check if stationary
            trecbr += 1
            dtbr = 0

        # next time step
        t += 1
        t_Vector.append(Tc)
        F_max.append(Fcorr.max() / dt)

    stop = timeit.default_timer()
    t = stop - start
    print('\nCPU time')
    print(t)

    # save last time step
    if TIME[-1] < p.T:
        print('t = {:.3f} s / {:.1f} s'.format(Tc, p.T))
        W, W_, TIME = write_results(W, W_, w, w_, TIME, Tc)

    return W, W_, TIME
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# GRAPHICS
# --------------------------------------------------------------------------- #
def plot_results(space, b, W, W_, TIME, ghost_cells=False, title=''):

    H1 = W_[:, :, 0]
    H2 = W_[:, :, 2]
    Q1 = W[:, :, 1]
    Q2 = W[:, :, 3]
    H = b + H1 + H2

    fig = plt.figure()
    plt.plot(space.x_uk, b, color='k')
    plt.plot(space.x_uk, b + H2[0, :], 'C0')
    plt.plot(space.x_uk, b + H2[0, :] + H1[0, :], 'C0')
    for i in range(1, W.shape[0]-2):
        plt.plot(space.x_uk, b + H2[i, :], color=[0.8, 0.8, 0.8])
        plt.plot(space.x_uk, b + H2[i, :] + H1[i, :], color=[0.8, 0.8, 0.8])
    plt.plot(space.x_uk, b + H2[-2, :], '-o', color='C3')
    plt.plot(space.x_uk, b + H2[-2, :] + H1[-2, :], '-o', color='C3')
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    # fig.savefig(f'Fig_change_Q{-Q1[-1, -1]:.0f}_H{H[-1, 0]*100:.0f}.png',
    #             bbox_inches="tight")

    max_depth = b.min()*1.1
    fig, ax = plt.subplots()
    plt.fill_between(space.x_uk, b + H2[-1, :] + H1[-1, :], max_depth,
                     color='lightcyan', label='freshwater')
    plt.fill_between(space.x_uk, b + H2[-1, :], max_depth,
                     color='C0', label='seawater')
    plt.fill_between(space.x_uk, b, max_depth,
                     color='lightgray', label='channel bed')
    plt.plot(space.x_uk, b,
             linewidth=1, color=[.1, .1, .1])
    plt.plot(space.x_uk,  b + H2[-1, :] + H1[-1, :],
             linewidth=0.5, color=[.1, .1, .1])
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)')
    plt.xlim([space.x_uk[0], space.x[-1]])
    plt.ylim([max_depth, H[-1, :].max() + 1])
    plt.title(title)
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    fig.savefig(f'Fig_profile_Q{-Q1[-1, -1]:.0f}_H{H[-1, 0]*100:.0f}.png',
                bbox_inches="tight")

    fig = plt.figure()
    plt.plot(Q1[:, 0])
    plt.title('Q1, mouth')
    plt.xlabel('time')
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    # fig.savefig(f'Fig_Q1mout_Q{-Q1[-1, -1]:.0f}_H{H[-1, 0]*100:.0f}.png',
    #             bbox_inches="tight")

    fig = plt.figure()
    plt.plot(Q2[:, 0])
    plt.title('Q2, mouth')
    plt.xlabel('time')
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    # fig.savefig(f'Fig_Q2mout_Q{-Q1[-1, -1]:.0f}_H{H[-1, 0]*100:.0f}.png',
    #             bbox_inches="tight")

    fig = plt.figure()
    plt.plot(Q1[-2, :])
    plt.title('Q1, final')
    plt.xlabel('length')
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    # fig.savefig(f'Fig_Q1up_Q{-Q1[-1, -1]:.0f}_H{H[-1, 0]*100:.0f}.png',
    #             bbox_inches="tight")

    fig = plt.figure()
    plt.plot(Q2[-2, :])
    plt.title('Q2, final')
    plt.xlabel('length')
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    # fig.savefig(f'Fig_Q2up_Q{-Q1[-1, -1]:.0f}_H{H[-1, 0]*100:.0f}.png',
    #             bbox_inches="tight")

# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# SAVE RESULTS
# --------------------------------------------------------------------------- #
def save_results(space, b, W, W_, TIME, ghost_cells=False):
    if os.path.isdir('RESULTS'):
        os.chdir('RESULTS')
    else:
        os.mkdir('RESULTS')
        os.chdir('RESULTS')
    every = 1
    np.savetxt('x.csv',
               space.x_uk[::every], fmt='%.6e', delimiter=', ')
    np.savetxt('b.csv',
               b[::every], fmt='%.6e', delimiter=', ')
    np.savetxt('H1.csv',
               W_[:, ::every, 0], fmt='%.6e', delimiter=', ')
    np.savetxt('H2.csv',
               W_[:, ::every, 2], fmt='%.6e', delimiter=', ')
    np.savetxt('Q1.csv',
               W[:, ::every, 1], fmt='%.6e', delimiter=', ')
    np.savetxt('Q2.csv',
               W[:, ::every, 3], fmt='%.6e', delimiter=', ')
    np.savetxt('U1.csv',
               W_[:, ::every, 1], fmt='%.6e', delimiter=', ')
    np.savetxt('U2.csv',
               W_[:, ::every, 3], fmt='%.6e', delimiter=', ')
    np.savetxt('TIME.csv',
               TIME, fmt='%.9e', delimiter=', ')

    # save last time step as initial condition
    np.savetxt('initial.txt', [W[-1, :, 0], W[-1, :, 1],
                               W[-1, :, 2], W[-1, :, 3]],
               fmt='%.9e')

    os.chdir('..')
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# CASE FOLDER
# --------------------------------------------------------------------------- #
if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    raise ValueError('expected only one argument, the name of the case folder')
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# MODEL SETUP
# --------------------------------------------------------------------------- #
minB, dz = 10, 0.05
min_dubina = 5e-2
rfile = 'r.txt'
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# MAIN PART OF THE CODE
# --------------------------------------------------------------------------- #
os.chdir(fname)
for input_file in glob.glob('input*.txt'):

    print(f'case: {fname}')

    print(f'input file: {input_file}')

    print('loading inputs...')
    p = load_inputs(input_file)

    print('setting spatial domain...')
    space = set_spatial_domain(p)

    print('preparing geometry functions...')
    geo_mat, b = get_geometry(p.Lstart, p.Lend, p.dx, dz=dz, minB=minB,
                              plot=False)

    print('loading initial condition...')
    w, w_, sig, o = load_init_cond(space.x_uk, geo_mat, b, p,
                                    plot_initial=True)

    print('loading boundary conditions...')
    BC_types, BC, r_series = load_boun_cond(p)

    os.chdir('..')
    print('STARTING THE COMPUTATIONS...')
    W, W_, TIME = time_stepping(p, r_series, space, geo_mat, b, w, w_,
                                sig, o, BC_types, BC,
                                min_dubina=min_dubina,
                                two_layers=p.twolayers)

    os.chdir(fname)
    plot_results(space, b, W, W_, TIME, True, fname)

    print('Saving results...')
    save_results(space, b, W, W_, TIME, True)

    print('Simulation completed!')

os.chdir('..')
# --------------------------------------------------------------------------- #
