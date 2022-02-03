# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:37:56 2019

@author: Nino Krvavica
"""

import os
import math
import numpy as np
import matplotlib.pyplot  as plt
from scipy import optimize


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


def f_critical(interface, surface, Q1, Q2, geo_mat, r, g, idx):
    A2 = interp_geom(interface, geo_mat.yi, geo_mat.A[:, idx])
    A1 = interp_geom(surface, geo_mat.yi, geo_mat.A[:, idx]) - A2
    sig1 = interp_geom(surface, geo_mat.yi, geo_mat.B[:, idx])
    sig3 = interp_geom(interface, geo_mat.yi, geo_mat.B[:, idx])
    sig2 = 1 / ((1-r) / sig3 + r / sig1)
    Fd1 = Q1**2 * sig1 * sig3 / (g * (1 - r) * A1**3 * sig2)
    if A2 == 0:
        Fd2 = 0
    else:
        Fd2 = Q2**2 * sig3 / (g * (1 - r) * A2**3)
    return (Fd1 + Fd2 - (1-r) * sig2 / sig3 * Fd1 * Fd2) - 1


def find_H_critical(surface, Q1, Q2, sig1, bottom, geo_mat, r, g, idx=0):
    # first guess
    sig1 = sig1*0.7
    A1_0 = (Q1**2 * sig1 / (g * (1-r)))**(1/3)
    A_tot = interp_geom(surface, geo_mat.yi, geo_mat.A[:, idx])
    A2_0 = max(A_tot - A1_0, 0)
    a = interp_geom(A2_0, geo_mat.ai, geo_mat.D[:, idx])
    return optimize.newton(f_critical, a, args=(surface, Q1, Q2,
                                                geo_mat, r, g, idx))


def eq(y_new, f_new, y, f, dx):
    return y_new - y - 0.5 * dx * (f + f_new)


def y1_and_f1(H, i, geo_mat, Q, wfric, ks, li, g, r):
    H1, H2 = H
    A1 = (interp_geom(H1, geo_mat.yi, geo_mat.A[:, i])
          - interp_geom(H2, geo_mat.yi, geo_mat.A[:, i]))
    O1 = (interp_geom(H1, geo_mat.yi, geo_mat.O[:, i])
          - interp_geom(H2, geo_mat.yi, geo_mat.O[:, i]))
    R = A1 / O1
    sig3 = interp_geom(H2, geo_mat.yi, geo_mat.B[:, i])
    y1 = Q**2 / (2 * g * A1**2) + H1
    if wfric == 1:
        lw = g * ks**2 * R**(-1/3)
    elif wfric == 2:
        Re = Q / A1 * R / 1e-6
        lw = 1/32 * (np.log10(ks / (12 * R) + 1.95 / Re**0.9))**(-2)
    else:
        lw = 0
    f1 = Q**2 / (g * A1**3) * (lw * O1 + li * sig3)  # i interfacial i wall friction u gornjem sloju
    return y1, f1


def y2_and_f2(H, i, geo_mat, Q, wfric, lw, li, g, r):
    H1, H2 = H
    A2 = interp_geom(H2, geo_mat.yi, geo_mat.A[:, i])
    A1 = interp_geom(H1, geo_mat.yi, geo_mat.A[:, i]) - A2
    sig3 = interp_geom(H2, geo_mat.yi, geo_mat.B[:, i])
    y2 = (1 - r) * H2 + r * H1
    f2 = - Q**2 / (g * A1**2 * A2) * li * r * sig3
    return y2, f2


def get_Jacobian2(H, y1, f1, y2, f2, step, dx, *kwargs):
    J = np.zeros((2, 2))

    H[0] = H[0] + step*0.5
    y_new1, f_new1 = y1_and_f1(H, *kwargs)
    J00_1 = eq(y_new1, f_new1, y1, f1, dx)
    y_new2, f_new2 = y2_and_f2(H, *kwargs)
    J10_1 = eq(y_new2, f_new2, y2, f2, dx)

    H[0] = H[0] - step
    y_new1, f_new1 = y1_and_f1(H, *kwargs)
    J00_2 = eq(y_new1, f_new1, y1, f1, dx)
    y_new2, f_new2 = y2_and_f2(H, *kwargs)
    J10_2 = eq(y_new2, f_new2, y2, f2, dx)

    H[0] = H[0] + step*0.5

    H[1] = H[1] + step*0.5
    y_new1, f_new1 = y1_and_f1(H, *kwargs)
    J01_1 = eq(y_new1, f_new1, y1, f1, dx)
    y_new2, f_new2 = y2_and_f2(H, *kwargs)
    J11_1 = eq(y_new2, f_new2, y2, f2, dx)

    H[1] = H[1] - step
    y_new1, f_new1 = y1_and_f1(H, *kwargs)
    J01_2 = eq(y_new1, f_new1, y1, f1, dx)
    y_new2, f_new2 = y2_and_f2(H, *kwargs)
    J11_2 = eq(y_new2, f_new2, y2, f2, dx)

    J[0, 0] = (J00_1 - J00_2) / step
    J[1, 0] = (J10_1 - J10_2) / step
    J[0, 1] = (J01_1 - J01_2) / step
    J[1, 1] = (J11_1 - J11_2) / step

    return J


def y_and_f(H, i, geo_mat, Q, wfric, ks, g):
    A = interp_geom(H, geo_mat.yi, geo_mat.A[:, i])
    if A == 0:
        return None, None
    O = interp_geom(H, geo_mat.yi, geo_mat.O[:, i])
    y = Q**2 / (2 * g * A**2) + H
    R = A / O
    if wfric == 1:
        f = ks**2 * Q**2 / A**2 * R**(-4/3)
    elif wfric == 2:
        Re = Q / A * R / 1e-6
        fw = 1/32 * (np.log10(ks / (12 * R) + 1.95 / Re**0.9))**(-2)
        f  = fw / g * Q**2 / A**2 / R
    else:
        f = 0
    return y, f

def get_Jacobian(H, y, f, step, dx, *kwargs):
    H = H + step*0.5
    y_new, f_new = y_and_f(H, *kwargs)
    J1 = eq(y_new, f_new, y, f, dx)
    H = H - step
    y_new, f_new = y_and_f(H, *kwargs)
    J2 = eq(y_new, f_new, y, f, dx)
    return (J1 - J2) / step


def froude(i, h1, h2, geom_f, Q, g, r):
    A2 = geom_f.h2a[i](h2)
    A1 = geom_f.h2a[i](h1) - A2
    sig1 = geom_f.h2s[i](h1)
    sig3 = geom_f.h2s[i](h2)
    sig2 = 1 / (1/sig1 + 1/sig3)
    Fd = Q / np.sqrt(g * (1-r) * A1**3 * sig2 / sig3 / sig1)
    return Fd

def river_froude(i, h1, geom_f, Q, g, r):
    A1 = geom_f.h2a[i](h1)
    sig1 = geom_f.h2s[i](h1)
    Fd = Q / np.sqrt(g * (1-r) * A1**3 / sig1)
    return Fd


def implicit_trap_2L(p, space, geo_mat, b, Hcr, level, Q, step=1e-9, tol=1e-6):
    h1 = np.zeros_like(b)
    h2 = np.zeros_like(b)
    h1[0] = level
    h2[0] = Hcr
    H_new = np.array([level, Hcr])
    fi = 1e-7 * np.abs(Q) + 7e-4    # fi_4
    y1, f1 = y1_and_f1(H_new, 0, geo_mat, Q, p.wfric, p.ks, fi, p.g, p.r)
    y2, f2 = y2_and_f2(H_new, 0, geo_mat, Q, p.wfric, p.ks, fi, p.g, p.r)
    for i in range(1, space.ndx+2):
        dh = 2 * p.dx * tol
        br = 0
        while np.abs(dh / p.dx).max() > tol and br < 50:
            br += 1
            y1_new, f1_new = y1_and_f1(H_new, i, geo_mat, Q, p.wfric, p.ks, fi,
                                       p.g, p.r)
            y2_new, f2_new = y2_and_f2(H_new, i, geo_mat, Q, p.wfric, p.ks, fi,
                                       p.g, p.r)
            J = get_Jacobian2(H_new, y1, f1, y2, f2, step, p.dx, i, geo_mat,
                              Q, p.wfric, p.ks, fi, p.g, p.r)
            R1 = eq(y1_new, f1_new, y1, f1, p.dx)
            R2 = eq(y2_new, f2_new, y2, f2, p.dx)
            R = np.array([R1, R2])
            dh = np.linalg.solve(J, R)
            H_new = H_new - dh
        if np.isnan(H_new).any() or (H_new < b[i]).any():
            break
        h1[i] = H_new[0]
        h2[i] = H_new[1]
        y1, f1 = y1_new, f1_new
        y2, f2 = y2_new, f2_new
    if i < space.ndx + 1:
        # compute Froude number
        print('not finished, computing one-layer solutions...')
        j = i - 1
        H_new = h1[j]
        y, f = y_and_f(H_new, j, geo_mat, Q, p.wfric, p.ks, p.g)
        for i in range(j + 1, space.ndx+2):
            dh = 2 * p.dx * tol
            while np.abs(dh / p.dx).max() > tol:
                y_new, f_new = y_and_f(H_new, i, geo_mat, Q, p.wfric, p.ks, p.g)
                J = get_Jacobian(H_new, y, f, step, p.dx, i, geo_mat,
                                 Q, p.wfric, p.ks, p.g)
                R = eq(y_new, f_new, y, f, p.dx)
                if J==0:
                    break
                dh = R / J
                H_new = H_new - dh
            h1[i] = H_new
            h2[i] = b[i]
            y, f = y_new, f_new
    return h1, h2


def implicit_trap_1L(p, space, geo_mat, b, Hcr, level, Q, step=1e-9, tol=1e-6):
    h1 = np.zeros_like(b)
    h2 = b
    h1[0] = level
    H_new = h1[0]
    y, f = y_and_f(H_new, 0, geo_mat, Q, p.wfric, p.ks, p.g)
    for i in range(1, space.ndx+2):
        H_new = max(h1[i-1], h2[i-1])
        print(i-1, H_new)
        dh = 2 * p.dx * tol
        while np.abs(dh / p.dx).max() > tol:
            y_new, f_new = y_and_f(H_new, i, geo_mat, Q, p.wfric, p.ks, p.g)
            if not y_new:
                H_new = H_new + 0.05
                continue
            J = get_Jacobian(H_new, y, f, step, p.dx, i, geo_mat,
                             Q, p.wfric, p.ks, p.g)
            R = eq(y_new, f_new, y, f, p.dx)
            if J == 0:
                break
            dh = R / J
            H_new = H_new - dh
        h1[i] = H_new
        h2[i] = b[i]
        y, f = y_new, f_new
    return h1, h2



def plotaj_uzduzni(x, b, h2, h1, Q, level, letter=None, is_large=True):
    x = x / 1000
    max_depth = b.min()*1.1
    fig, ax = plt.subplots()
    plt.fill_between(x, h1, max_depth,
                     color='lightcyan', label='freshwater')
    plt.fill_between(x, h2, max_depth,
                     color='C0', label='seawater')
    plt.fill_between(x, b, max_depth,
                     color='lightgray', label='channel bed')
    plt.plot(x, b,
             linewidth=1, color=[.1, .1, .1])
    plt.plot(x, h1,
             linewidth=0.5, color=[.1, .1, .1])
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)')
    if letter:
        plt.text(-0.12, 1.0, letter, fontsize=14, transform=ax.transAxes)
    # Save figure
    if is_large:
        fig.set_size_inches(6, 3)
    else:
        fig.set_size_inches(6, 3)
    fig.tight_layout()
    fig.savefig(f'Fig_stac_Q{Q:.0f}_H{level*100:.0f}.png', bbox_inches="tight")



def main(p, space, geo_mat, b, level, Q, two_layer=True, plot=True):
    print('setting initial condition...')
    sig1 = interp_geom(level, geo_mat.yi, geo_mat.B[:, 0])
    if two_layer:
        Hcr = find_H_critical(level, Q*1.05, 0, sig1, b, geo_mat, p.r, p.g)
        Hcr = max(Hcr, b[0])
        if Hcr == b[0]:
            two_layer = False
    else:
        Hcr = b[0]
    print('computing two-layer steady solution...')
    if two_layer:
        h1, h2 = implicit_trap_2L(p, space, geo_mat, b, Hcr, level, Q)
    else:
        h1, h2 = implicit_trap_1L(p, space, geo_mat, b, Hcr, level, Q)
    if plot:
        plotaj_uzduzni(space.x_uk, b, h2, h1, Q, level)
    return h1, h2
