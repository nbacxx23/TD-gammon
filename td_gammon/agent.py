import numpy as numpy
from numpy import random

class agent(object):

    def __init__(self,player,nnModel):
        self.player=player
        self.name='TD-gammon '+player
        self.model=nnModel


    def sel_best(self,combinations,gammon):
        """select the best combinations by evaluate the state value using the neuron network
        """
        v_best=0
        com_best=None
        
        #TODO: deepcopy the result of the combination
        #Question: how to implement the TD(lambda) valuation here
        
        for comb in combinations:
            """
            TODO: code to modify :
            _,_,_=gammon.doMoves(comb,self.player)
            input= gammon.toFeatures(gammon.opp(self.player))
            value=self.model.evaluate(input)
            """
            temp=gammon.clone()
            temp.doCombination(self.player,comb)
            input= temp.toFeatures(temp.opp(self.player))
            value=self.model.evaluate(input)
            if self.player==gammon.players[1]:
                value=1.0-value
            if value>v_best:
                v_best=value
                com_best=comb

        return com_best
