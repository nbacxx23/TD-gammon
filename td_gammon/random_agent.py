import numpy as numpy
from numpy import random

class randomAgent(object):
    
    def __init__(self,player):
        self.player=player
        self.name='Random Agent '+player
    
    
    def sel_best(self,combinations,gammon):
        """select a random combination
            """
        comb=combinations[random.randint(0,len(combinations))]
        return comb
