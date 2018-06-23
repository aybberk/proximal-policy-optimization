import tensorflow as tf
import numpy as np
import gym
from time import sleep

tf.reset_default_graph()



pd = np.random.randn(8)
pd -= np.min(pd)
pd /= np.sum(pd)