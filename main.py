from Gammon import *
from random_agent import *
#-------first code for testing the gammon schelon
#-----run python main.py to see simple game between two random players
Game=Gammon()

agents=[]
agent1=randomAgent('o')
agent2=randomAgent('x')
agents.append(agent1)
agents.append(agent2)

Game.startGame(agents)
