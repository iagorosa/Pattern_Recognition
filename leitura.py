#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 12:15:33 2019

@author: iagorosa
""" 

import scipy.io as scio
from PIL import Image as pil
import numpy as np

mat = scio.loadmat('Yale_32x32.mat')
m = mat['fea']
np.savetxt("Yale_32x32.csv", m, delimiter=',', fmt="%d")

#m0 = m[100].reshape(32,32)
#img = pil.fromarray(m0, 'L')
#img.show()

np.savetxt("Yale_32x32.csv", m, delimiter=',', fmt="%d")