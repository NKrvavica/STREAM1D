# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:03:50 2019

@author: Nino Krvavica
"""

import numpy as np


def compute_fi(equation, Q1, Q2, A1, A2, u1, u2, h1, h2, r, g=9.8):
    if equation == 1:
        return 2e-5 * np.abs(Q1 - Q2) / 0.1
    if equation == 2:
        return 2e-4 * np.abs(Q1 - Q2) / 0.5
    if equation == 3:
        return 2e-4 * np.abs(Q1 - Q2) / 0.3
    if equation == 4:
        return 2e-4 * np.abs(Q1 - Q2) / 0.9
    if equation == 5:
        return 2e-4 * np.abs(Q1 - Q2) / 1.0
    if equation == 6:
        return 2e-4 * np.abs(Q1 - Q2) / 9.6
    if equation == 7:
        return 2e-4 * np.abs(Q1 - Q2) / 1.9
    if equation == 8:
        return 2e-4 * np.abs(Q1 - Q2) / 4.9
    if equation == 9:
        return 2e-4 * np.abs(Q1 - Q2) / 128.4
    if equation == 10:
        return 2e-4 * np.abs(Q1 - Q2) / 5.9
