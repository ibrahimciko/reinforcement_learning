#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:32:12 2020

@author: ibrahim
"""

from numpy.random import binomial
from numpy.random import multinomial
from numpy import where

class Environment():
    
    
    def __init__(self,bandits,arms,is_life_style=None, population_prob=None):
        """
        Environment Class where bandit and arms interact. Bandit chooses an arm and observes the outcome

        Parameters
        ----------
        bandits: list
            list of bandit objects exmpl [e_greedy, thompson, ...]
            
        arms: list
            list of arm objects
            
        is_life_style: Boolean
        
        population_prob: list 
            list of probabilities for population distribution
            
        

        Attributes
        ----------
        true_probs: list of true reward probabilities of each arm
        
        optimal_index: Integer arm index whose true probability is the highest
        
        optimal_index_per_life_style:
            
        optimal_prob:
            
        optimal_prob_per_life_style:
        
        reward_history: Nested List of bandits' reward history
        
        choice history: Nested List of bandits' choice history
        
        num_chosen_arms: Nested list of bandits' chosen arms
        
        regrests: Nested list of bandits regrets
        
        data: dict where bandit.name is the key, and its attributes are the values

        """
        
        self.bandits = bandits
        self.arms = arms
        self.reward_history = [bandit.reward_history for bandit in self.bandits]
        self.choice_history = [bandit.choice_history for bandit in self.bandits]
        self.num_chosen_arms = [bandit.num_chosen_arms for bandit in self.bandits]
        self.regrets = [bandit.regret for bandit in self.bandits]
        self.data = {bandit.name: bandit.data for bandit in self.bandits}
        
        assert(is_life_style == arms[0].is_life_style)
        self.is_life_style = is_life_style
        
        if self.is_life_style:
           assert(population_prob)
           self.population_prob = population_prob 
           self.num_life_style = len(self.population_prob)          
           self.true_probs = [arm.true_life_style_prob for arm in self.arms]   
           self.optimal_index_per_life_style = []
           self.optimal_prob_per_life_style = []
           for i in range(self.num_life_style):
               optimal_value = -1
               optimal_arm_index = 0
               for j in range(len(self.arms)):
                   if self.arms[j].true_life_style_prob[i] >= optimal_value:
                       optimal_value = self.arms[j].true_life_style_prob[i]
                       optimal_arm_index = j
               self.optimal_index_per_life_style.append(optimal_arm_index)
               self.optimal_prob_per_life_style.append(optimal_value)
            
        else:
            self.true_probs = [arm.true_prob for arm in self.arms]  
            self.optimal_prob = max(self.true_probs)
            self.optimal_index = self.true_probs.index(self.optimal_prob)

       
    def sample_ctr(self,prob):
        """
        Samples a 1-0 from Bernoulli given the probability

        Parameters
        ----------
        prob : float
            Probability of success for bernoulli trial

        Returns
        -------
        ctr : Integer - 1 or 0 (success or failure)

        """
        ctr = binomial(1,prob)
        return ctr
    
    def sample_life_style(self):
        """
        Samples from multinomial distribution whose probabilities are given by
        self.population_prob.

        Returns
        -------
        ind - Int: Index of the life style group that is sampled

        """
        assert(self.is_life_style)
        
        ind = where(multinomial(1,self.population_prob))[0][0]
        return ind

        
    
    def interaction(self):
        """
        bandit and arms interacts and bandit chooses an arm. Parameters then updated

        """
        for i in range(len(self.bandits)):
            
        
            #bandit's turn
            choice = self.bandits[i].choose_arm()
            
            #God's turn
            if self.is_life_style:
                life_style = self.sample_life_style()
                prob = self.true_probs[choice][life_style]
                ctr = self.sample_ctr(prob)
                self.bandits[i].regret.append(int(self.optimal_index_per_life_style[life_style] != choice))
            
            else:
                ctr = self.sample_ctr(self.true_probs[choice])
                self.bandits[i].regret.append(int(self.optimal_index != choice))
                
            self.bandits[i].observe_outcome(ctr,choice)
            
       
    def clear_interaction(self):
        """
        Method to clear bandit data, and start from the initial belief - default_probs
        It clears the following attributes:
            
                a-) reward_history
                b-) num_chosen_arms
                c-) choice_history
                e-) regret
                f-) data
    
        """
        for i in range(len(self.bandits)):
            self.bandits[i].delete_history()

# debug           
if __name__ == "__main__":     
    from thompson import ThompsonBandit
    from arm import Arm
    K = 2
    g1 = ThompsonBandit("thompson", default_priors = [[1,1] for i in range(K)]) #flat
    a1 = Arm("a1",is_life_style=True,true_life_style_prob=[0.9,0.1])
    a2 = Arm("a2",is_life_style=True,true_life_style_prob=[0.1,0.9])    
    env = Environment([g1],[a1,a2],True,[0.8,0.1])
    
    print(env.optimal_index)
    print(env.optimal_index_per_life_style)
    print(env.num_life_style)
    print(env.optimal_prob_per_life_style)

    if env.population_prob:
        print("lifestyle")
        
    #interact
    env.interaction()
    env.is_life_style
    print(g1.regret)
    print(env.choice_history)
    print(env.reward_history)
    print(env.optimal_index)
    print(env.regrets)
    env.bandits[0].perceived_probs
    env.clear_interaction()
      
