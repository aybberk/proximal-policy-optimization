from PolicyCNN import PolicyCNN
from ValueCNN import ValueCNN
from RolloutWorker import RolloutWorker
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Hyperparameters import LEARNING_RATE, NUM_MINIBATCHES, EPSILON, NUM_WORKERS, \
T, C1, C2, ENV_ID, INPUT_SHAPE, ACTIONS, LR_ANNEALING_RATE, \
NUM_EPOCHS, EPSILON_ANNEALING_RATE, CLIP_VALUE_LOSS, MAX_GRAD_NORM

class PPOAgent:
    def __init__(self):
        self._build_graph()
        self.train_history = {
                "average_values": [],
                "average_returns": [],
                "average_entropy": [],
                "frames_trained": 0,
                }
        self.worker = RolloutWorker(self, ENV_ID, NUM_WORKERS, T)

    def _build_graph(self):
        self.policy_net = PolicyCNN("policy_net", 
                                        input_shape = INPUT_SHAPE,
                                        num_actions = len(ACTIONS))
        
        self.value_net = ValueCNN("value_net",
                                  input_shape = INPUT_SHAPE)

        
        self.ACTIONS = self.policy_net.actions_ph

        
        self.ADVANTAGES = tf.placeholder(dtype=tf.float32, 
                                            shape = [None],
                                            name = "advantages_ph")
    
        self.OLD_ACTION_PROBS = tf.placeholder(dtype=tf.float32, 
                                            shape = [None],
                                            name = "old_action_probs_ph")
        
        self.OLD_ACTION_LOG_PROBS = tf.placeholder(dtype=tf.float32, 
                                            shape = [None],
                                            name = "old_action_log_probs_ph")
        
        
        self.OLD_VALUES = tf.placeholder(dtype=tf.float32, 
                                            shape = [None],
                                            name = "old_values_ph")
        
        self.TARGET_VALUES = tf.placeholder(dtype=tf.float32,
                                            shape=[None],
                                            name="values_ph")
  
        self.mean_ADV, self.std_ADV = tf.nn.moments(self.ADVANTAGES, axes=0)
        self.advantages_normalized = (self.ADVANTAGES - self.mean_ADV) / (self.std_ADV + 1e-8)
        
        self.r = tf.exp(self.policy_net.actions_log_prob - self.OLD_ACTION_LOG_PROBS)
        
        
        self.clipped_r = tf.clip_by_value(self.r, 1-EPSILON, 1+EPSILON)
        
        self.surr = self.r * self.advantages_normalized 
        self.surr_clip = self.clipped_r * self.advantages_normalized
        
        self.L_CLIP_batch = tf.minimum(self.surr, self.surr_clip)
        self.L_CLIP = -tf.reduce_mean(self.L_CLIP_batch)

        self.v_out = self.value_net.values
        self.v_out_clipped = self.OLD_VALUES + tf.clip_by_value(self.v_out - self.OLD_VALUES, -EPSILON, EPSILON)
        self.L_V_batch = tf.square(self.v_out - self.TARGET_VALUES)
        self.L_V_clip_batch = tf.square(self.v_out_clipped - self.TARGET_VALUES)
        
        self.L_VF_clip   = 0.5 * tf.reduce_mean(tf.maximum(self.L_V_batch, self.L_V_clip_batch))
        self.L_VF_noclip = 0.5 * tf.reduce_mean(self.L_V_batch)
        
        if CLIP_VALUE_LOSS:
            self.L_VF = self.L_VF_clip
        else:
            self.L_VF = self.L_VF_noclip
        
        
        self.L_ENT = -tf.reduce_mean(self.policy_net.entropy, 
                                     axis = 0)
        
        
        self.L_VFxC1 = self.L_VF * C1
        self.L_ENTxC2 = self.L_ENT * C2
                
        
        self.loss = self.L_CLIP + self.L_VFxC1 + self.L_ENTxC2
       
        self.learning_rate = tf.Variable(initial_value = LEARNING_RATE, 
                                         dtype = tf.float32, 
                                         trainable = False)
        
        self.epsilon = tf.Variable(initial_value = EPSILON, 
                                   dtype = tf.float32, 
                                   trainable = False)
        
        self.anneal_learning_rate = tf.assign_sub(self.learning_rate, 
                                                  LR_ANNEALING_RATE)
        self.anneal_epsilon = tf.assign_sub(self.epsilon, 
                                            EPSILON_ANNEALING_RATE)
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-5)
        self.params = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.params)
        
        if MAX_GRAD_NORM is not None:
            self.grads_clipped, self.grad_norm = tf.clip_by_global_norm(self.grads, MAX_GRAD_NORM)
        
        self.grads_clipped_applyable = list(zip(self.grads_clipped, self.params))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=1e-5)
        self.optimize_op = self.optimizer.apply_gradients(self.grads_clipped_applyable)
        
#        self.gradients, self.variables = zip(*self.optimizer.compute_gradients(self.loss))
#        self.gradients_clipped, self.global_norm = tf.clip_by_global_norm(self.gradients, 0.5)
#        self.optimize_op = self.optimizer.apply_gradients(zip(self.gradients_clipped, self.variables))
    
    
    def train_step(self, sess):
      
        states, old_probs, old_log_probs, old_action_probs, old_action_log_probs,\
        actions, rewards, returns, advantages, old_values\
        = self.worker.rollout(sess)
        self.train_history['average_returns'].append(returns.mean())
        self.train_history['average_values'].append(old_values.mean())
        self.train_history['average_entropy'].append(-np.mean(np.sum(old_probs * old_log_probs, axis=1)))
        self.train_history['frames_trained'] += states.shape[0] * 4
       
        
        print(self.train_history['frames_trained'])
        
        l_clip, l_vf, l_ent = sess.run([self.L_CLIP, self.L_VFxC1, self.L_ENTxC2], 
                           feed_dict= {
                                   self.policy_net.states_ph: states,
                                   self.value_net.states_ph: states,
                                   self.ACTIONS: actions,
                                   self.ADVANTAGES: advantages,
                                   self.TARGET_VALUES: returns,
                                   self.OLD_ACTION_PROBS: old_action_probs,
                                   self.OLD_ACTION_LOG_PROBS: old_action_log_probs,
                                   self.OLD_VALUES: old_values,
                                   }
                           )
        
        for epoch in range(NUM_EPOCHS):
            dataset_size = states.shape[0]
            batch_size = dataset_size // NUM_MINIBATCHES
            random_indices = np.arange(dataset_size)
            np.random.shuffle(random_indices)
            
            for n in range(NUM_MINIBATCHES):
                batch_indices = random_indices[n * batch_size: n * batch_size + batch_size]
                states_batch = states[batch_indices]
                old_action_log_probs_batch = old_action_log_probs[batch_indices]
                actions_batch = actions[batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                old_values_batch = old_values[batch_indices]
         
                _ = sess.run([self.optimize_op, self.grads, self.grads_clipped, self.grad_norm], 
                                   feed_dict= {
                                           self.policy_net.states_ph: states_batch,
                                           self.value_net.states_ph: states_batch,
                                           self.ACTIONS: actions_batch,
                                           self.ADVANTAGES: advantages_batch,
                                           self.TARGET_VALUES: returns_batch,
                                           self.OLD_ACTION_LOG_PROBS: old_action_log_probs_batch,
                                           self.OLD_VALUES: old_values_batch,
                                           }
                                   )
                
        l_clip2, l_vf2, l_ent2 = sess.run([self.L_CLIP, self.L_VFxC1, self.L_ENTxC2], 
                           feed_dict= {
                                   self.policy_net.states_ph: states,
                                   self.value_net.states_ph: states,
                                   self.ACTIONS: actions,
                                   self.ADVANTAGES: advantages,
                                   self.TARGET_VALUES: returns,
                                   self.OLD_ACTION_PROBS: old_action_probs,
                                   self.OLD_ACTION_LOG_PROBS: old_action_log_probs,
                                   self.OLD_VALUES: old_values,
                                   }
                           )

                
        print("Loss before: {: .6f} {:.6f} {:.6f}".format(l_clip, l_vf, l_ent))
        print("Loss after : {: .6f} {:.6f} {:.6f}".format(l_clip2, l_vf2, l_ent2))
                
        self.anneal(sess)
        

        
    def anneal(self, sess):
        a,b = sess.run([self.anneal_learning_rate, self.anneal_epsilon])
        
    def reset_optimizer_state(self, sess):
        pass
        
