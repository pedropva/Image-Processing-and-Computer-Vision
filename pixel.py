# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:53:51 2018

@author: pedropva
"""

class pixel:
    """A simple example class"""
    coords =  []
    color = []
    
    def f(self):
        return 'hello world'
    
    def __init__(self,x,y,color):
    self.coords = [x,y]
    self.color = color
    
    def translate(self):
        return "REE"