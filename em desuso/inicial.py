#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:20:39 2019

@author: iagorosa
"""

import numpy as np
from PIL import Image as pil
     
# leitura imagem
path = 'yalefaces/'
img = pil.open(path+"subject01.centerlight")
img.show() 

# reconstrução da imagem
arr = np.array(img)
img2 = pil.fromarray(arr, 'L')
img2.show()
