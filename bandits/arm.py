#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:31:13 2020

@author: ibrahim
"""


class Arm():
    
    def __init__(self,name,true_prob=None, is_life_style=None, true_life_style_prob=None):
        """
        Arm - An image, news, video to be chosen by people

        Parameters
        ----------
        name : String
        
        true_prob : list of float, optional
            True probabilities of rewards bandits don't know. The default is None.    
            
        life_style : Bool, optional
            Whether probabilities are conditional on features such as life style. The default is False.

            
        true_life_style_prob : list of float, optional
            True reward probabilities for each cluster or life style. The default is None.

        """
        self.name = name   
        self.is_life_style = is_life_style
        
        if not is_life_style:
            self.true_prob = true_prob
        else:
            self.num_life_style = len(true_life_style_prob)
            self.true_life_style_prob = true_life_style_prob
            
    def __repr__(self):
        return self.name
    

    # Debugging
    if __name__ == "__main__":
        a1 = Arm("a1",life_style=True,true_life_style_prob=[0.8,0.7,0.1,0])
        a2 = Arm("a2",life_style=True,true_life_style_prob=[0.1,0.4,0.9,0.4])
        a1_t = a1.true_life_style_prob
        a2_t = a2.true_life_style_prob
        
        a3 = Arm("a3",life_style=False,true_prob = 0.7)
        a3.life_style
        a3.true_life_style_prob
        

