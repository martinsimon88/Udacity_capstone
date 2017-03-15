#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import itertools as it
import OpenVSPGame as VSP
import PolicyTest1 as PT1


#Test connection with OpenVSPGame
print ("Is connection with OpenVSPGame?  "+VSP.testVSP)
#Test connection with PolicyTest1
print ("Is connection with PolicyTest1?  "+PT1.testPT1)



# --------------- EXPERIMENTS ---------------

showcase=False
'''
# Test settings
scaled_resolution = (48, 64)
batch_size = 64
sequence_length = batch_size
e_start = 1.0

# Resume
start_from = 0

# Super simple basic
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

n = VSP.buttons
actions = [list(a) for a in it.product([0, 1], repeat=n)]

observation_history = 0

print("Creating learner")
learner = PT1.Learner(available_actions_count=len(actions),
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
                  OutputParameters=scaled_resolution,
                  replay_memory_size=replay_memory_size,
                  start_from=start_from,
                  load_model=load_model,
                  p_decay=p_decay,
                  e_start=e_start,
                  model_loadfile=script_dir+"/tf/"+model_name+".ckpt",
                  model_savefile=script_dir+"/tf/"+model_name+"_out.ckpt")
'''

Area = 40;
Sweep = 45;
actions = [Area, Sweep];




if not showcase:
    print("Training learner")
    #learner.learn(VSPGame, actions)
    print("initiate game")
    VSP.start_game(1, actions);
else:
    learner.play(VSPGame, actions, episodes_to_watch=episodes_to_watch)


#if __name__ == '__main__':
#    VSP.start_game(1, actions)