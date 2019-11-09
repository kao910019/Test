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
from tqdm import tqdm

# hyper parameters
render = False
num_episode = 3000
max_steps = 1000

class Network(object):
    def __init__(self, session, num_features, num_actions):
        self.session = session
        self.num_actions = num_actions
        self.num_features = num_features
        
        # GAMMA:
        # The bigger the more attention is paid to the experience.
        # The smaller the more attention is paid to the benefits.
        self.gamma = 0.9
        
        # Critic must learn faster than Actor.
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        
        with tf.variable_scope("placeholder"):
            self.state = tf.placeholder(tf.float32, [None, num_features], "state")
            self.action = tf.placeholder(tf.int32, None, "action")
            # Temporal Difference Error
            self.a_td_error = tf.placeholder(tf.float32, None, "td_error")
            self.v_next = tf.placeholder(tf.float32, [1, 1], "v_next")
            self.reward = tf.placeholder(tf.float32, None, 'reward')
            
        self.build_critic()
        self.build_actor()
    
    def build_critic(self):
        with tf.variable_scope('Critic'):
            output = tf.layers.dense(self.state, 20, activation=tf.nn.relu)
            self.v = tf.layers.dense(output, 1)

            self.c_td_error = self.reward + self.gamma * self.v_next - self.v
            self.loss = tf.square(self.c_td_error)    # TD_error = (r+gamma*V_next) - V_eval
            self.train_critic_op = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.loss)
    
    def build_actor(self):
        with tf.variable_scope('Actor'):
            output = tf.layers.dense(self.state, 20, activation=tf.nn.relu)
            self.action_prob = tf.layers.dense(output, self.num_actions, activation=tf.nn.softmax)

            log_prob = tf.log(self.action_prob[0, self.action])
            self.exp_v = tf.reduce_mean(log_prob * self.a_td_error)  # advantage (TD_error) guided loss
            self.train_actor_op = tf.train.AdamOptimizer(self.actor_learning_rate).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
    
    def update(self, last_state, state, reward, action):
        last_state = np.reshape(last_state, (1, self.num_features))
        state = np.reshape(state, (1, self.num_features))
        # Update Critic
        v_next = self.session.run(self.v, feed_dict = {self.state: state})
        td_error, _ = self.session.run([self.c_td_error, self.train_critic_op],
                                       feed_dict = {self.state: last_state, 
                                                    self.v_next: v_next, 
                                                    self.reward: reward})
        # Update Actor
        exp_v, _ = self.session.run([self.exp_v, self.train_actor_op], 
                                    feed_dict = {self.state: last_state, 
                                                 self.action: action,
                                                 self.a_td_error: td_error})
    
    def choose_action(self, state):
        state = np.reshape(state, (1, self.num_features))
        prob = self.session.run(self.action_prob, {self.state: state})   # get probabilities for all actions
        return np.random.choice(range(len(prob[0])), p=prob[0])   # return a int

env = gym.make('CartPole-v0')
env = env.unwrapped

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

with tf.Session() as sess:
    with sess.graph.as_default():
        network = Network(sess, num_features, num_actions)
        
    sess.run(tf.global_variables_initializer())
    
    
    process_bar = tqdm(range(num_episode))
    for episode in process_bar:
        last_state = env.reset()
        reward_sum = 0
        for _ in range(max_steps):
            # very 300 step show the robot
            if render:
                env.render()
            
            action = network.choose_action(last_state)
            
            state, reward, done, info = env.step(action)
            
            if done: reward = -20.0
            
            network.update(last_state, state, reward, action)
            
            last_state = state
            reward_sum += reward
            
            if done:
                process_bar.set_description("Reward = {}".format(reward_sum))
                render = True if reward_sum >= 200 else False
                break
    
    print("# Training Complete.")
    last_state = env.reset()
    while True:
        env.render()
        action = network.choose_action(last_state)
        state, reward, done, info = env.step(action)
        last_state = state
        if done:
            last_state = env.reset()
        
