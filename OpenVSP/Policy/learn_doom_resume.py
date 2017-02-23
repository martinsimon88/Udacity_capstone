#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import math
import experience_replay as er
import os

import cv2

class Learner:

    def __init__(self,
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
                 reward_exploration=False,
                 reward_shooting=False,
                 resolution=(30, 45),
                 sequence_length=10,
                 observation_history=4,
                 death_match=False,
                 model_loadfile="/tmp/model.ckpt",
                 model_savefile="/tmp/model.ckpt",
                 start_from=0,
                 save_model=True,
                 load_model=False):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.learning_steps_per_epoch = learning_steps_per_epoch
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self.frame_repeat = frame_repeat
        self.p_decay = p_decay
        self.e_start = e_start
        self.resolution = resolution
        self.available_actions_count = available_actions_count
        self.model_savefile = model_savefile
        self.save_model = save_model
        self.load_model = load_model
        self.death_match = death_match
        self.reward_exploration = reward_exploration
        self.sequence_length = sequence_length
        self.observation_history = observation_history
        self.update_every = update_every
        self.start_from = start_from
        self.model_loadfile = model_loadfile
        self.reward_shooting = reward_shooting

        # Positions traversed during an episode
        self.positions = []

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

        kills_before = game.get_game_variable(GameVariable.KILLCOUNT)
        ammo_before = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        armor_before = game.get_game_variable(GameVariable.ARMOR)
        reward = game.make_action(actions[a], self.frame_repeat)
        ammo_after = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        armor_after = game.get_game_variable(GameVariable.ARMOR)
        kills_after = game.get_game_variable(GameVariable.KILLCOUNT)
        ammo_gained = (ammo_after - ammo_before)
        selected_weapon = game.get_game_variable(GameVariable.SELECTED_WEAPON)

        if self.reward_shooting:
            enemies = self.get_enemy_count(game)
            shoot_reward = abs(ammo_gained) * enemies   # Bonus for shooting if enemies on screen
            if selected_weapon == 2:
                pistol_bonus=30

            else:
                pistol_bonus=0
            if enemies == 0:
                shoot_reward = ammo_gained
            kill_reward = (kills_after - kills_before) * 20
            reward = shoot_reward + kill_reward + pistol_bonus
            if game.is_player_dead():
                reward -= 10
            if ammo_gained > 0:
                reward += 5
            reward -= 1  # Life sucks
        elif reward_exploration:
            reward = self.exploration_reward(game)
        elif self.death_match:
            if game.is_player_dead():
                reward = -100
            else:
                reward = (kills_after - kills_before) * 500 \
                         + (armor_after - armor_before) \
                         + ammo_gained

        #if reward > 0:
            #print("Reward: " + str(reward))

        isterminal = game.is_episode_finished()
        s2 = self.preprocess(game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        if learning_step % self.update_every == 0:
            self.learn_from_memory()

        return reward

    def get_enemy_count(self, game):
        state = game.get_state()
        enemies = 0
        for l in state.labels:
            #print("Object id: " + str(l.object_id) + " object name: " + l.object_name + " label: " + str(l.value))
            if l.object_id != 0 and l.object_name == "DoomPlayer":
                enemies += 1
        return enemies

    def get_position(self, game):
        return (game.get_game_variable(GameVariable.PLAYER_POSITION_X), game.get_game_variable(GameVariable.PLAYER_POSITION_Y))

    def distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) / 100

    def exploration_reward(self, game):

        pos = self.get_position(game)

        if len(self.positions) < 0:
            self.positions.append(pos)
            return 0

        weighted_sum = 0
        t = 0
        last_pos = ()
        for p in reversed(self.positions):
            if t == 0:
                last_pos = p
                weighted_sum += self.distance(pos, last_pos)
            else:
                new_distance = self.distance(pos, p)
                old_distance = self.distance(last_pos, p)
                diff = new_distance - old_distance
                weighted_diff = diff * (self.p_decay**t)
                weighted_sum += weighted_diff
            t += 1

        self.positions.append(pos)

        return weighted_sum


    def preprocess(self, img):
        """ Converts and down-samples the input image. """
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        return img

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


class DoomServer:

    def __init__(self, screen_resolution, config_file_path, deathmatch=False, bots=7, visual=False, async=True):
        self.screen_resolution = screen_resolution
        self.deathmatch = deathmatch
        self.bots = bots
        self.visual = visual
        self.async = async
        self.config_file_path = config_file_path

    def new_game(self):
        game = DoomGame()
        game.load_config(self.config_file_path)
        game.set_window_visible(self.visual)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(self.screen_resolution)
        game.set_labels_buffer_enabled(True)
        if self.deathmatch:
            #self.game.set_doom_map("map01")  # Limited deathmatch.
            game.set_doom_map("map02")  # Full deathmatch.
            # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
            game.add_game_args("-host 1 -deathmatch +timelimit 2.0 "
                               "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
            # Name your agent and select color
            # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            game.add_game_args("+name AI +colorset 0")

        if self.async:
            game.set_mode(Mode.ASYNC_PLAYER)
        else:
            game.set_mode(Mode.PLAYER)

        game.init()

        self.start_game(game)

        return game

    def addBots(self, game):
        game.send_game_command("removebots")
        for i in range(self.bots):
            game.send_game_command("addbot")

    def start_game(self, game):
        started = False
        while not started:
            try:
                game.new_episode()
            except:
                print("Failed to restart ViZDoom!")
            else:
                started = True

        if self.deathmatch:
            self.addBots(game)


# --------------- EXPERIMENTS ---------------

# Test settings
visual = False
async = False
screen_resolution = ScreenResolution.RES_320X240
scaled_resolution = (48, 64)
batch_size = 64
sequence_length = batch_size
e_start = 1.0
reward_shooting = False

# Override these if used
p_decay = 1
bots = 7
observation_history = 0

# Resume
start_from = 0

# Super simple basic
'''
hidden_nodes = 1
conv1_filters = 1
conv2_filters = 1
replay_memory_size = 10
frame_repeat = 12
learning_steps_per_epoch = 100
test_episodes_per_epoch = 1
reward_exploration = False
epochs = 10
model_name = "super_simple_basic"
death_match = False
config = "../config/simpler_basic.cfg"
'''

# Simple basic
'''
hidden_nodes = 128
conv1_filters = 8
conv2_filters = 8
replay_memory_size = 10000
frame_repeat = 12
learning_steps_per_epoch = 1000
test_episodes_per_epoch = 10
reward_exploration = False
epochs = 20
model_name = "simple_basic"
death_match = False
config = "../config/simpler_basic.cfg"
'''

# Simple advanced
'''
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 1000000
frame_repeat = 4
learning_steps_per_epoch = 2000
test_episodes_per_epoch = 10
reward_exploration = False
epochs = 20
model_name = "simple_adv"
death_match = False
config = "../config/simpler_adv.cfg"
'''

# Simple exploration
'''
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 1000000
frame_repeat = 4
learning_steps_per_epoch = 2000
test_episodes_per_epoch = 10
reward_exploration = True
epochs = 50
model_name = "simple_exploration"
death_match = False
config = "../config/simpler_adv_expl.cfg"
p_decay = 0.90
'''

# Deathmatch simple exploration
'''
hidden_nodes = 12
conv1_filters = 4
conv2_filters = 8
replay_memory_size = 1000
frame_repeat = 4
learning_steps_per_epoch = 5000
test_episodes_per_epoch = 10
reward_exploration = True
epochs = 10
model_name = "deathmatch_exploration_simple_no_bots"
death_match = True
#bots = 0
config = "../config/cig_train_expl.cfg"
p_decay = 0.90
'''

# Deathmatch exploration
'''
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 100000
frame_repeat = 4
learning_steps_per_epoch = 10000
test_episodes_per_epoch = 10
reward_exploration = True
epochs = 200
model_name = "deathmatch_exploration_no_bots_2"
death_match = True
bots = 0
config = "../config/cig_train_expl.cfg"
p_decay = 0.95
'''

# Deathmatch from exploration
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 100000
frame_repeat = 4
learning_steps_per_epoch = 10000
test_episodes_per_epoch = 10
reward_exploration = False
reward_shooting = True
epochs = 300
model_name = "deathmatch_exploration_no_bots_2"
death_match = True
bots = 7
config = "../config/cig_train.cfg"
e_start = 0.50
load_model = True

# Deathmatch killing from deathmatch shooting
'''
hidden_nodes = 512
conv1_filters = 32
conv2_filters = 64
replay_memory_size = 100000
frame_repeat = 4
learning_steps_per_epoch = 10000
test_episodes_per_epoch = 10
reward_exploration = False
reward_shooting = False
epochs = 400
model_name = "deathmatch_shooting_reward"
death_match = True
bots = 7
config = "../config/cig_train.cfg"
e_start = 0.25
load_model = True
'''

# ---------------- SHOWCASE ----------------
showcase = False
episodes_to_watch = 10

# Uncomment these
'''
async = True
visual = True
showcase = True
'''

# ------------------------------------------------------------------
server = DoomServer(screen_resolution=screen_resolution,
                    config_file_path=config,
                    deathmatch=death_match,
                    visual=visual,
                    async=async,
                    bots=bots)

print("Starting game to get actions.")
game = server.new_game()
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
game.close()
print("Game closed again")

script_dir = os.path.dirname(os.path.abspath(__file__)) #<-- absolute dir the script is in
print("Script path="+script_dir)

print("Creating learner")

learner = Learner(available_actions_count=len(actions),
                  frame_repeat=frame_repeat,
                  hidden_nodes=hidden_nodes,
                  conv1_filters=conv1_filters,
                  conv2_filters=conv2_filters,
                  epochs=epochs,
                  batch_size=batch_size,
                  sequence_length=sequence_length,
                  observation_history=observation_history,
                  learning_steps_per_epoch=learning_steps_per_epoch,
                  test_episodes_per_epoch=test_episodes_per_epoch,
                  reward_exploration=reward_exploration,
                  reward_shooting=reward_shooting,
                  resolution=scaled_resolution,
                  replay_memory_size=replay_memory_size,
                  start_from=start_from,
                  load_model=load_model,
                  p_decay=p_decay,
                  e_start=e_start,
                  death_match=death_match,
                  model_loadfile=script_dir+"/tf/"+model_name+".ckpt",
                  model_savefile=script_dir+"/tf/"+model_name+"_out.ckpt")

if not showcase:
    print("Training learner")
    learner.learn(server, actions)
else:
    learner.play(server, actions, episodes_to_watch=episodes_to_watch)
