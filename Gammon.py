import numpy as np
from numpy import random
import os
from copy import deepcopy

#To define a class as the backgammon game
#--------essential functions----------
#---startGame(self,agents,show=False): start a new game given two agent players until the end
#---toFeatures(self,player): extract the state features as the NN input
#---allCombinations(self,player,rolls): all possible move combinations given rolls

class Gammon:

    #parameters of games
    PLAYERS=['o','x']
    DEFAULT=[['o',[0],2],['o',[11,18],5],['o',[16],3],['x',[23],2],['x',[7],3],['x',[5,12],5]]
    
    #HIT='hit'
    #END='end'
    
    def __init__(self, state=None,players=None,hitState=None,endState=None):
        self.numCols=24
        self.winner=None
        if state:
            #use deepcopy, else the origin could not remain unchanged
            self.state=deepcopy(state)
            self.players = deepcopy(players)
            self.hitState=deepcopy(hitState)
            self.endState=deepcopy(endState)
        else:
            self.reset()
            self.players=Gammon.PLAYERS
            self.hitState={}
            self.endState={}
            for player in self.players:
                self.hitState[player]=0
                self.endState[player]=0


    #the state by default of backgammon
    def reset(self,begin=DEFAULT):
        self.state=[[] for col in range(self.numCols)]
        for _token,_cols,_num in begin:
            for _col in _cols:
                self.state[_col]=[_token]*_num

    #roll two dies each round
    def rolling(self):
        return random.randint(1,6,size=2)

    #clone a gammon game for evaluating the state value
    def clone(self):
        return Gammon(self.state,self.players,self.hitState,self.endState)
    
    def startGame(self,agents,show=False):
        #decide the first turn to move
        turn=random.randint(0,1)
        STEP=0
        while not self.is_end():
            self.next(agents[turn],show=show)
            turn=0 if turn else 1
            STEP+=1
        #----code for debug-----comment later
        print str(STEP) +' steps, Winnier: '+ self.winner
        return self.winner
    
    #next step change the player
    def next(self,agent,show=False):
        player=agent.player
        rolls=self.rolling()
        #----code for debug-----comment later
        print 'player :'+ player
        print rolls
        #change perspective if neccesary
        if player!=self.players[0]:
            self.exchange()
        Combs=self.allCombinations(player,rolls)
        if Combs:
            bestComb=agent.sel_best(Combs,self)
            self.doCombination(player,bestComb)
            #----code for debug-----comment later
            print bestComb

    #to tell if the game is over
    def is_end(self):
        for player in self.players:
            if self.endState[player]==15:
                self.winner=player
                return True
        return False

    #get the other player
    def opp(self,player):
        for p in self.players:
            if p!=player:
                return p

    #verify if a move is valid
    def is_valid(self, player, move):
        if player in self.state[move[0]]:
            if (self.opp(player) in self.state[move[1]]) and len(self.state[move[1]])>1:
                return False
            else:
                return True
        return False

    #exchange the direction of the board
    #TODO: double check if it is logic
    def exchange(self):
        self.state.reverse()
        self.players.reverse()

    #verify if an end is valid
    def is_end_valid(self,player):
        num=0
        """
        change the direction every time, make the player 0-23
        
        if player==self.player[0]:
            for _col in range(self.numCols-6,self.numCols):
                if len(self.state[_col])>0 and self.state[_col][0]==player:
                    num+=len(self.state[_col])
        else:
            for _col in range(0,6):
                if len(self.state[_col])>0 and self.state[_col][0]==player:
                    num+=len(self.state[_col])
        """
        for _col in range(self.numCols-6,self.numCols):
            if len(self.state[_col])>0 and self.state[_col][0]==player:
                num+=len(self.state[_col])

        if num==(15-self.endState[player]):
            return True
        else:
            return False

    #assume that the condition for removing pieces is reached
    def is_remove_valid(self,player,col1,roll):
        if col1<self.numCols-6:
            return False
        if len(self.state[col1])==0:
            return False
        if self.state[col1][0]==player:
            if (col1+roll)==self.numCols:
                return True
            if (col1+roll)>self.numCols:
                for _col in range(self.numCols-6,self.numCols-roll):
                    if player in self.state[_col]:
                        return False
                return True
        return False

    #verify if a move is valid, unhit and remove are not considered here
    def is_move_valid(self,player,col1,col2):
        if col2<=col1 or col1<0 or col2>self.numCols-1:
            return False
        if len(self.state[col1])==0:
            return False
        if self.state[col1][0]!=player:
            return False
        if len(self.state[col2])<=1:
            return True
        if self.state[col2][0]==player:
            return True
        return False


    #verify if a roll valid to unhit the piece (use roll-1)
    def is_unhit_valid(self,player,roll):
        if len(self.state[roll-1])<=1:
            return True
        if len(self.state[roll-1])>1 and self.state[roll-1][0]==player:
            return True
        return  False

    #conduct the given move for the player, assuming that the given move is valid
    def doMove(self, player, move):
        col1,col2=move
        if col1=='hit':
            self.hitState[player]-=1
        else:
            del self.state[col1][-1]
        if col2=='end':
            self.endState[player]+=1
        else:
            if len(self.state[col2])==1 and self.state[col2][0]==self.opp(player):
                self.hitState[self.opp(player)]+=1
                del self.state[col2][-1]
            self.state[col2].append(player)
                             
    #conduct the given combination of moves for the player, assuming that the given move is valid
    def doCombination(self,player,comb):
        for move in comb:
            self.doMove(player,move)


    #give all possible moves given a roll length
    def allMoves(self,player,roll):
        """check if hit onboard first, if so and unhit is not valid, return None
        """
        Moves=[]
        if self.hitState[player]>0:
            if self.is_unhit_valid(player,roll):
                Moves.append(('hit',roll-1))
                return Moves
            else:
                return None
        for _col in range(0,self.numCols-roll):
            if self.is_move_valid(player,_col,_col+roll):
                Moves.append((_col,_col+roll))
        if self.is_end_valid(player):
            for _col in range(self.numCols-roll,self.numCols):
                if self.is_remove_valid(player,_col,roll):
                    Moves.append((_col,'end'))
        #Question: is the blocked situation logic?
        if len(Moves)==0:
            return None
        return Moves

                             
    #give all possible combinations with the roll
    #QUESTION: Shall we remove the combination of different orders?
    def allCombinations(self,player,rolls):
        #list of combinations
        Comb=[]
        r1,r2=rolls
        if r1==r2:
            k=4
            moves=self.allMoves(player,r1)
            if moves:
                k-=1
                for move in moves:
                    Comb.append([move])
            else:
                return None
            while k>0:
                tempComb=deepcopy(Comb)
                for moves in tempComb:
                    temp=self.clone()
                    for move in moves:
                        temp.doMove(player,move)
                    postMoves=temp.allMoves(player,r1)
                    if postMoves:
                        for postMove in postMoves:
                            Comb.append(moves+[postMove])
                        Comb.remove(moves)
                k-=1
            return Comb if Comb else None

        for _r1,_r2 in [(r1,r2),(r2,r1)]:
            moves=self.allMoves(player,_r1)
            if moves:
                for move in moves:
                    temp=self.clone()
                    temp.doMove(player,move)
                    postMoves=temp.allMoves(player,_r2)
                    if postMoves:
                        for postMove in postMoves:
                            Comb.append([move,postMove])
                    else:
                        Comb.append([move])
        return Comb if Comb else None

    #extract feature of the state for NN
    def toFeatures(self,player):
        features=[]
        #extract for each player
        for token in self.players:
            for _col in self.state:
                #four dimensions for each column
                feat=[0.0]*4
                num=len(_col)
                if (token in _col):
                    for i in range(min(num,3)):
                        feat[i]=1.
                    if num>3:
                        feat[3]+=(num-3)/2
                features+=feat
            #number of hit pieces
            features.append(self.hitState[token]/2.)
            #number of end pieces, 15 in total
            features.append(self.endState[token]/15.)
        #two dimension representing the turn of the game
        if player==self.players[0]:
            features+=[1.,0.]
        else:
            features+=[0.,1.]
        return features







