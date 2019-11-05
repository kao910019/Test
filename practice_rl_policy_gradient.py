# -*- coding: utf-8 -*-
"""
        # install gym on server
        
        git clone https://github.com/openai/gym
        cd gym
        pip install -e .
        pip install -e '.[all]'
        
"""
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

# hyper parameters
num_iters = 3000
max_steps = 100000
num_dimensions = 4
learning_rate = 1e-3

gamma = 0.99 # discount factor for reward

# Create gym env
env = gym.make("CartPole-v0")

#placeholder
state_placeholder = tf.placeholder(tf.float32, [None, num_dimensions])
actions_placeholder = tf.placeholder(tf.int32, [None, 1])
advantages_placeholder = tf.placeholder(tf.float32, [None, 1])
        
def Network(state, actions, advantages):
    with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
        # 1 layer fc
        output = tf.layers.dense(state, 10, activation=tf.nn.relu)
        prob = tf.nn.softmax(tf.layers.dense(output, 2), axis=-1)
        
        prob_given_state = tf.reduce_sum(-tf.log(prob) * tf.one_hot(actions, 2), axis=0)
        loss = tf.reduce_mean(prob_given_state * advantages)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
        return prob, train_op

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r

def choose_action(prob):
    action = np.random.choice(range(len(prob)), p=prob)  # select action w.r.t the actions prob
    return action

with tf.Session() as sess:
    with sess.graph.as_default():
        # Draw your neural network here.
        prob, train_op = Network(state_placeholder, actions_placeholder, advantages_placeholder)
    
    #init variables
    sess.run(tf.global_variables_initializer())

    reward_sum = 0
    
    process_bar = tqdm(range(num_iters))
    for episode in process_bar:
        # reset our env
        observation = env.reset()
        feed_states, feed_actions, feed_reward = [], [], []
        reward_sum = 0
        
        for _ in range(max_steps):
            # very 300 step show the robot
            if episode % 300 == 0:
                env.render()
            
            state = np.reshape(observation, (1, num_dimensions))
            # put env state into network
            action_prob = sess.run(prob, feed_dict={state_placeholder: state})
            # select an action based on policy gradient
            action = choose_action(action_prob[0]) 
            # run on env and get reward and next step state
            observation, reward, done, info = env.step(action)
            
            feed_states.append(state)
            feed_actions.append(action)
            feed_reward.append(reward)
            
            # step the environment and get new measurements
            reward_sum += reward
            
            if done: # an episode finished
                process_bar.set_description("Reward = {}".format(reward_sum))
                
                feed_advantages = discount_rewards(feed_reward) # compute discounted and normalized rewards
                
                state = np.reshape(np.array(feed_states), [-1, num_dimensions])
                advantages = np.reshape(np.array(feed_advantages), [-1, 1])
                action = np.reshape(np.array(feed_actions), [-1, 1])
                
                sess.run(train_op, feed_dict={state_placeholder: state, 
                                              advantages_placeholder: advantages, 
                                              actions_placeholder: action})
                break
    
    print("# Training Complete.")
    observation = env.reset()
    while True:
        env.render()
        state = np.reshape(observation, (1, num_dimensions))
        action_prob = sess.run(prob, feed_dict={state_placeholder: state})
        action = choose_action(action_prob[0])
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
