# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:28:56 2017

@author: ee4yo
"""

import cv2
from tqdm import tqdm      # barra de porcentagem.
import os                  # Lida com diretorios

DIR = 'data/test'
DIRS = 'corta/'

for imagem in tqdm(os.listdir(DIR)):

    label = imagem
    path = os.path.join(DIR,imagem)
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    
    tamanho = img.shape[0]    
    valor = round(tamanho*0.3)
    img = img[valor:tamanho-valor,valor:tamanho-valor]
    
    cv2.imwrite(DIRS + label, img, [cv2.IMWRITE_JPEG_QUALITY, 100])