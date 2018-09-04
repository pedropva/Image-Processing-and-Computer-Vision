# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:18:09 2018

@author: pedropva
"""

def multMatriz(mat1, mat2):
    matRes = []
    for i in range(0,len(mat1)):     #for(i = 0; i < mat1.length; i++)
        res = []
        for j in range(0,len(mat2[0])):#for(j = 0; j < mat2[0].length; j++)
            s = 0
            for k in range(0,len(mat2)):   #for(k = 0; k < mat2.length; k++)
                s += mat1[i][k] * mat2[k][j]
            res.append(s)
        matRes.append(res)
    return matRes
