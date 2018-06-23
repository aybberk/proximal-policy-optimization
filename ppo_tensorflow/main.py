import tensorflow as tf
import numpy as np
import gym
from PPOAgent import PPOAgent

tf.reset_default_graph()
agent = PPOAgent()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for n in range(3000):
        agent.train_step(sess)