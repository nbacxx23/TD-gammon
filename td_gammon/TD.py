from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf

from Gammon import *
from random_agent import *
from agent import *

# helper to initialize a weight and bias variable
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
    return W, b

# helper to create a dense, fully-connected layer
def dense_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(shape)
        return activation(tf.matmul(x, W) + b, name='activation')

class Model(object):
    def __init__(self, sess, model_path,  checkpoint_path, restore=False):
        self.model_path = model_path
        
        self.checkpoint_path = checkpoint_path

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay
        lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step, \
            30000, 0.96, staircase=True), name='lambda')

        # learning rate decay
        learning_rate = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step, \
            40000, 0.96, staircase=True), name='alpha')

       

        # describe network size
        layer_size_input = 198
        layer_size_hidden = 60
        layer_size_output = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, layer_size_input], name='x')
        self.Value_next = tf.placeholder('float', [1, layer_size_output], name='Value_next')

        # build network arch. (just 2 layers with sigmoid activation)
        value_tmp = dense_layer(self.x, [layer_size_input, layer_size_hidden], tf.sigmoid, name='layer1')
        self.Value = dense_layer(value_tmp, [layer_size_hidden, layer_size_output], tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.scalar_summary('Value_next', tf.reduce_sum(self.Value_next))
        tf.scalar_summary('Value', tf.reduce_sum(self.Value))

        # delta = Value_next - Value
        delta_op = tf.reduce_sum(self.Value_next - self.Value, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.Value_next - self.Value), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.Value_next), tf.round(self.Value)), dtype='float'), name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            one_game_step = tf.Variable(tf.constant(0.0), name='one_game_step', trainable=False)
            one_game_step_op = one_game_step.assign_add(1.0)
            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            loss_sum_op = loss_sum.assign_add(loss_op)
            loss_avg_op = loss_sum / tf.maximum(one_game_step, 1.0)


            # reset per-game monitoring variables
            game_step_reset_op = one_game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.Value, tvars)


        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)

                # grad with trace = alpha * delta * e
                grad_trace = learning_rate * delta_op * trace_op

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            one_game_step_op,
            loss_sum_op,
            loss_avg_op,
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')


        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.initialize_all_variables())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def evaluate(self, x):
        return self.sess.run(self.Value, feed_dict={ self.x: x })

    def test(self, episodes=100):
        agents = [agent('o', self), randomAgent('x')]
        winners = {}
        winners['o']=0
        winners['x']=0
        for episode in range(episodes):
            game = Gammon()
            winner = game.startGame(agents)
            winners[winner] += 1

            winners_total = winners['o']+winners['x']
            print("[Episode %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode, \
                agents[0].name, agents[0].player, \
                agents[1].name, agents[1].player, \
                winners['o'], winners['x'], winners_total, \
                (winners['o'] / winners_total) * 100.0))



    def train(self):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_gammon.pb', as_text=False)

        agents = [agent('o', self), agent('x', self)]

        validation_interval = 50
        episodes = 10

        for episode in range(episodes):
            
            game = Gammon()
            turn = random.randint(0, 1)

            x = game.toFeatures(agents[turn].player)

            STEP = 0
            while not game.is_end():
                game.next(agents[turn])
                turn = (turn + 1) % 2

                x_next = game.toFeatures(agents[turn].player)
                Value_next = self.evaluate(x_next)
                self.sess.run(self.train_op, feed_dict={ self.x: x, self.Value_next: Value_next })
                game.exchange()
                x = game.toFeatures(agents[turn].player)
                STEP += 1
            
            

            _, global_step,  _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.reset_op
            ], feed_dict={ self.x: x, self.Value_next: np.array([[0]], dtype='float') })


            _, global_step,  _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.reset_op
            ], feed_dict={ self.x: x_next, self.Value_next: np.array([[1]], dtype='float') })
           

            print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, game.winner, STEP))
            self.saver.save(self.sess, self.checkpoint_path + 'checkpoint', global_step=global_step)
        self.test()
        
 
