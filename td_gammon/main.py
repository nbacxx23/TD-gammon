from Gammon import *
from random_agent import *
from agent import *
from TD import *
import tensorflow as tf

#-------first code for testing the gammon schelon
#-----run python main.py to see simple game between two random players
"""
Game=Gammon()

agents=[]
agent1=randomAgent('o')
agent2=randomAgent('x')
agents.append(agent1)
agents.append(agent2)

Game.startGame(agents)
"""
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, demand a recommendation of a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')
model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)



if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        model = Model(sess, model_path,  checkpoint_path, restore=FLAGS.restore)
        if FLAGS.test:
            model.test(episodes=100)
        elif FLAGS.play:
            model.play()
        else:
            model.train()