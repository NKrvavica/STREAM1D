# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:49:12 2019

@author: Nino Krvavica
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from scipy.interpolate import griddata, interp2d, interp1d
from collections import namedtuple


def load_geometry():
    # load corss-section profiles (given as an array (N, 3))
    width_points = pd.read_csv('width_profiles', delimiter='\t',
                               index_col=None, header=None)
    width_points.columns = ['x', 'z', 'y']
    width_points_sorted = width_points.sort_values(by='x')
    return width_points_sorted


def reverse_interpolate(yi, ai, A):
    A = A.T
    D = np.zeros((A.shape[0], len(ai)))
    for n in range(A.shape[0]):
        profile = A[n, :]
        idx = np.argmax(profile > 0)
        di = np.interp(ai, profile[idx-1:], yi[idx-1:])
        D[n, :] = di
    return D.T


def interp_geom(x, xi, yi):
    # water levels
    temp = (x - xi[0]) / (xi[-1] - xi[0]) * (len(xi) - 1)
    idx_down = np.floor(temp).astype(np.int32)
    idx_up = idx_down + 1
    if yi.ndim == 1:
        y_down = yi[idx_down]
        y_up = yi[idx_up]
    else:
        n = yi.shape[1]
        y_down = yi[idx_down, np.arange(n)]
        y_up = yi[idx_up, np.arange(n)]
    return y_down + (y_up - y_down) * (temp - idx_down)


#@profile
def get_geometry(Lstart, Lend, dl, dz=0.01, minB=1, plot=False):
    print('loading width profiles...')
    # load geometry
    width_points = load_geometry()

    # get minimum and maximum vertical coordinates
    zmin = width_points['z'].min() - 2 * dz
    zmax = width_points['z'].max() + 2 * dz

    # get bottom profile
    bottom_profile = []
    current_x = width_points['x'][0]
    while True:
        profile_mask = np.array(width_points['x'] == current_x).nonzero()[0]
        profile = width_points.iloc[profile_mask, :]
        profile = profile.sort_values(by='z')
        bottom_profile.append(np.array([current_x, profile.iloc[0, 1]]))
        if profile_mask[-1] == width_points.shape[0] - 1:
            break
        else:
            current_x = width_points.iloc[profile_mask[-1] + 1, 0]
    bottom_profile = np.array(bottom_profile)

    # strech all profiles over z (for interpolation)
    current_x = width_points['x'][0]
    width_array_z_streched = np.zeros((1, 3))
    while True:
        profile_mask = np.array(width_points['x'] == current_x).nonzero()[0]
        profile = width_points.iloc[profile_mask, :]
        profile = profile.sort_values(by='z')
        new_z = np.linspace(profile['z'].iloc[0], profile['z'].iloc[-1], 100)
        new_width = np.interp(new_z, profile['z'], profile['y'])
        z_stretched = np.linspace(zmin, zmax, 100)
        profile_z_streched = np.vstack((np.ones((len(new_z))) * current_x,
                                        z_stretched, new_width)).T
        width_array_z_streched = np.vstack((width_array_z_streched,
                                            profile_z_streched))
        if profile_mask[-1] == width_points.shape[0] - 1:
            break
        else:
            current_x = width_points.iloc[profile_mask[-1] + 1, 0]
    width_array_z_streched = np.delete(width_array_z_streched, 0, 0)

    print('interpolating...')
    # interpolate widths over a rectengular grid (For a given dl and dz)
    xi = np.arange(Lstart + dl/2, Lend, dl)
    yi = np.arange(zmin, zmax, dz)
    x, y = np.meshgrid(xi, yi, sparse=False, indexing='xy')
    B_z_stretched = griddata(width_array_z_streched[:, 0:2],
                             width_array_z_streched[:, 2], (x, y),
                             method='linear')

    # interpolate bottom profile
    new_bottom_profile = np.interp(xi, bottom_profile[:, 0],
                                   bottom_profile[:, 1])
    bottom = []
    for z in new_bottom_profile:
        idx = np.argwhere(yi <= z)[-1]
        bottom.append(yi[idx])
    bottom = np.array(bottom)
    bottom = np.vstack((xi, bottom[:, 0])).T

    # restore real depths for width profiles
    B = np.zeros_like(B_z_stretched)
    for n, profile in enumerate(B_z_stretched.T):
        min_z = bottom[n, 1]
        N = int((yi[-1] - min_z) / dz) + 1
        real_z = np.linspace(zmin, zmax, N)
        real_width = np.interp(real_z, yi, profile)
        real_width[real_width < minB] = minB
        B[-N:, n] = real_width
        if N + 1 <= len(profile):
            B[-N - 1, n] = minB

    print('computing cross-section properties...')
    # compute area and perimiter matrix from width matrix
    yi_half = yi[:-1] + dz/2
    A = []
    O = []
    for prof in B.T:
        B_mean = np.interp(yi_half, yi, prof)
        idx = np.argwhere(B_mean > 0)[0]
        B_mean[idx] = 0
        A_prof = np.insert(np.cumsum(B_mean * dz), 0, 0)
        A.append(A_prof)
        B_diff = np.diff(prof)
        dr = 2 * np.sqrt((0.5*B_diff)**2 + (dz)**2) * (A_prof[1:]!=0)
        dr[idx] = prof[np.argwhere(prof > 0)[0]]
        O_prof = np.insert(np.cumsum(dr), 0, 0)
        O.append(O_prof)
    A = np.array(A).T
    O = np.array(O).T

    # add ghost cells
    A = np.column_stack((A[:, 0], A, A[:, -1]))
    B = np.column_stack((B[:, 0], B, B[:, -1]))
    O = np.column_stack((O[:, 0], O, O[:, -1]))
    b = np.hstack((bottom[0, 1], bottom[:, 1], bottom[-1, 1]))

    print('inversing cross-section properties...')
    ai_korak = dz * max(minB, 10) / 10
    ai = np.arange(0, np.ceil(A.max()), ai_korak)
    D = reverse_interpolate(yi, ai, A)


    if plot:
        print('plotting geometry matrices...')
        # plot width, area and perimeter matrices
        plt.figure()
        c2 = plt.contourf(x, y, B[:, 1:-1],
                          levels=np.arange(minB-.1, np.max(B)+5, 1))
        plt.plot(bottom[:, 0], bottom[:, 1], c='C1')
        plt.title('Width (m)')
        plt.colorbar(c2)

        plt.figure()
        c2 = plt.contourf(x, y, A[:, 1:-1],
                          levels=np.arange(0, np.max(A)+5, 1))
        plt.plot(bottom[:, 0], bottom[:, 1], c='C1')
        plt.title('Area (m$^2$)')
        plt.colorbar(c2)

        plt.figure()
        c2 = plt.contourf(x, y, O[:, 1:-1],
                          levels=np.arange(0, np.max(O)+5, 1))
        plt.plot(bottom[:, 0], bottom[:, 1], c='C1')
        plt.title('Perimiter (m)')
        plt.colorbar(c2)

    print('returning named touple with geometry matrices...')
    GeometryFunction = namedtuple('GeometryFunction',
                                  ['A', 'B', 'O', 'D', 'yi', 'ai'])
    geo_mat = GeometryFunction(A, B, O, D, yi, ai)

    return geo_mat, b
