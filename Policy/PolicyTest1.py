#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import math
import experience_replay as er
import os
import OpenVSPGame as VSP
import cv2

testPT1= "yes"


#Test connection with OpenVSPGame
print ("Is connection with OpenVSPGame through PolicyTest1?  "+VSP.testVSP)


def Learner(self,
             available_actions_count,
             learning_rate=0.00025,
             discount_factor=0.99,
             epochs=20,
             hidden_nodes=4608,
             conv1_filters=32,
             conv2_filters=64,
             learning_steps_per_epoch=2000,
             replay_memory_size=10000,
             batch_size=64,
             test_episodes_per_epoch=2,
             frame_repeat=12,
             update_every=4,
             p_decay=0.95,
             e_start=1,
             reward_low_Cd=True,
             reward_high_Cl=True,
             resolution=(30, 45),
             sequence_length=10,
             observation_history=4,
             death_match=False,
             model_loadfile="/tmp/model.ckpt",
             model_savefile="/tmp/model.ckpt",
             start_from=0,
             save_model=True,
             load_model=False):


    # Create replay memory which will store the transitions
    print("Creating replay memory")
    self.memory = er.ReplayMemory(capacity=replay_memory_size, resolution=resolution)

    # Start TF session
    print("Starting session")
    self.session = tf.Session()

    print("Creating model")
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(self.resolution) + [1], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")

    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=conv1_filters, kernel_size=[6, 6], stride=[3, 3],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=conv2_filters, kernel_size=[3, 3], stride=[2, 2],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)

    #conv2_flat = tf.contrib.layers.DropoutLayer(conv2_flat, keep=0.5, name='dropout')

    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=hidden_nodes, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.constant_initializer(0.1))

    #fc1 = tf.contrib.layers.DropoutLayer(fc1, keep=0.5, name='dropout')

    #gru = tf.tensorlayer.RNNLayer(fc1, cell_fn=tf.nn.rnn_cell.GRUCell, n_hidden=128, n_steps=1, return_seq_2d=False)

    #gru = tf.contrib.layers.DropoutLayer(gru, keep=0.5, name='dropout')

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.available_actions_count, activation_fn=None,
                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                  biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.contrib.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _ = self.session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return self.session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return self.session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, self.resolution[0], self.resolution[1], 1]))[0]

    self.fn_learn = function_learn
    self.fn_get_q_values = function_get_q_values
    self.fn_get_best_action = function_simple_get_best_action

    print("Model created")

def learn_from_memory(self):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if self.memory.size > self.batch_size:
        s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

        q2 = np.max(self.fn_get_q_values(s2), axis=1)
        target_q = self.fn_get_q_values(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
        self.fn_learn(s1, target_q)

def exploration_rate(self, epoch, linear=False):
    """# Define exploration rate change over time"""
    start_eps = self.e_start
    end_eps = 0.1
    const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
    eps_decay_epochs = 0.6 * self.epochs  # 60% of learning time

    if linear:
        return max(start_eps - (epoch / self.epochs), end_eps)

    if epoch < const_eps_epochs:
        return start_eps
    elif epoch < eps_decay_epochs:
        # Linear decay
        return start_eps - (epoch - const_eps_epochs) / \
                           (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
    else:
        return end_eps

def perform_learning_step(self, game, actions, epoch, reward_exploration, learning_step):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    s1 = self.preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = self.exploration_rate(epoch, linear=False)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = self.fn_get_best_action(s1)

    #reward variables
    Cd
    Cl
    Cs


    if self.reward_low_Cd:
        Cd = readfromfileCd
    elif reward_high_Cl:
        reward = readfromfileCl*somevalue*Cd

    # Remember the transition that was just experienced.
    self.memory.add_transition(s1, a, s2, isterminal, reward)

    if learning_step % self.update_every == 0:
        self.learn_from_memory()

    return reward



def learn(self, server, actions):

    saver = tf.train.Saver()
    if self.load_model:
        print("Loading model from: ", self.model_loadfile)
        saver.restore(self.session, self.model_loadfile)
    else:
        init = tf.initialize_all_variables()
        self.session.run(init)

    print("Starting the training!")

    time_start = time()
    train_results = []
    test_results = []

    for epoch in range(self.epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        if epoch < self.start_from:
            continue;
        game = server.new_game()
        train_episodes_finished = 0
        train_scores = []
        print("Training...")
        eps = self.exploration_rate(epoch, linear=True)
        print("Epsilon: " + str(eps))
        self.positions = []
        score = 0
        for learning_step in trange(self.learning_steps_per_epoch):
            if game.is_player_dead():
                if self.reward_exploration:
                    train_scores.append(score)
                    train_episodes_finished += 1
                    score = 0
                self.positions = []
                game.respawn_player()

            if game.is_episode_finished() or learning_step+1 == self.learning_steps_per_epoch:
                if not self.reward_exploration:
                    if self.death_match:
                        score = game.get_game_variable(GameVariable.FRAGCOUNT)
                    else:
                        score = game.get_total_reward()
                train_scores.append(score)
                train_episodes_finished += 1
                self.positions = []
                score = 0
                game.close()
                game = server.new_game()

            reward = self.perform_learning_step(game, actions, epoch, self.reward_exploration, learning_step)

            if self.reward_exploration:
                score += reward

        print("%d training episodes played." % train_episodes_finished)

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        train_results.append(str(epoch) + " " + str(train_scores.mean()) + " " + str(train_scores.std()))

        print("\nTesting...")
        game.close()
        game = server.new_game()
        test_scores = []
        for test_episode in trange(self.test_episodes_per_epoch):
            self.positions = []
            score = 0
            while not game.is_episode_finished():
                state = self.preprocess(game.get_state().screen_buffer)
                best_action_index = self.fn_get_best_action(state)
                game.make_action(actions[best_action_index], self.frame_repeat)
                if self.reward_exploration:
                    score += self.exploration_reward(game)
            if not self.reward_exploration:
                if self.death_match:
                    score = game.get_game_variable(GameVariable.FRAGCOUNT)
                else:
                    score = game.get_total_reward()
            test_scores.append(score)
            game.close()
            game = server.new_game()

        test_scores = np.array(test_scores)
        print("Results: mean: %.1f±%.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
              "max: %.1f" % test_scores.max())

        test_results.append(str(epoch) + " " + str(test_scores.mean()) + " " + str(test_scores.std()))

        print("Saving the network weigths to:", self.model_savefile)
        saver.save(self.session, self.model_savefile)

        print("Saving the results...")
        t = time()
        with open("train_results_" + str(epoch) + ".txt", "w") as train_result_file:
            train_result_file.write(str(train_results))
        with open("test_results_" + str(epoch) + ".txt", "w") as test_result_file:
            test_result_file.write(str(test_results))

        print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    print("======================================")
    print("Training finished.")





def play(self, server, actions, episodes_to_watch=1):

    print("Loading model from: ", self.model_loadfile)
    saver = tf.train.Saver()
    saver.restore(self.session, self.model_loadfile)

    for _ in range(episodes_to_watch):
        game = server.new_game()
        score = 0
        while not game.is_episode_finished():
            state = self.preprocess(game.get_state().screen_buffer)
            best_action_index = self.fn_get_best_action(state)
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(self.frame_repeat):
                game.advance_action()

            if self.reward_exploration:
                score += self.exploration_reward(game)

        game.close()

        # Sleep between episodes
        sleep(1.0)
        if not self.reward_exploration:
            if self.death_match:
                score = game.get_game_variable(GameVariable.FRAGCOUNT)
            else:
                score = game.get_total_reward()
        print("Total score: ", score)
