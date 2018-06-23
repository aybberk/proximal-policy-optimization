import tensorflow as tf
import numpy as np

class PolicyCNN:
    def __init__(self, scope, input_shape, num_actions, is_trainable=True):
        with tf.variable_scope(scope):
            self.scope = scope
            self.states_ph = tf.placeholder(dtype=tf.float32, 
                                        shape=[None, *input_shape], 
                                        name="state")
            
            self.states_scaled = self.states_ph / 255.
            
            self.conv1 = tf.layers.conv2d(inputs=self.states_scaled, 
                                           filters=32, 
                                           kernel_size=[8, 8], 
                                           strides=4, 
                                           padding="valid", 
                                           activation=tf.nn.relu, 
                                           trainable=is_trainable,
                                           name="conv1")
            
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, 
                                           filters=64, 
                                           kernel_size=[4, 4], 
                                           strides=2, 
                                           padding="valid", 
                                           activation=tf.nn.relu, 
                                           trainable=is_trainable,
                                           name="conv2")
            
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, 
                                           filters=64, 
                                           kernel_size=[3, 3], 
                                           strides=1, 
                                           padding="valid", 
                                           activation=tf.nn.relu, 
                                           trainable=is_trainable,
                                           name="conv3")
            
            self.flatten = tf.layers.flatten(inputs=self.conv3,
                                             name="flatten")
            
            self.dense1 = tf.layers.dense(inputs=self.flatten,
                                          units=512,
                                          activation=tf.nn.relu,
                                          trainable=is_trainable,
                                          name="dense1")
            
            self.policy_logits = tf.layers.dense(inputs=self.dense1, 
                                          units=num_actions, 
                                          trainable=is_trainable)
            
            self.policy = tf.nn.softmax(self.policy_logits, axis = -1)
            self.log_policy = tf.nn.log_softmax(self.policy_logits)
            
            self.entropy = -tf.reduce_sum(self.policy * self.log_policy, axis = 1)
            
            self.actions_ph = tf.placeholder(dtype=tf.int32,
                                             shape=[None],
                                             name="actions_ph")

            #this is going to be used for selecting specific 
            #actions' probabilities
            self.reshaped_policy = tf.reshape(self.policy, [-1])
            self.reshaped_log_policy = tf.reshape(self.log_policy, [-1])
            #indices for selected actions at reshaped policy
            self.indices = tf.range(0, tf.shape(self.actions_ph)[0]) * tf.shape(self.policy)[1] + self.actions_ph
            
            
            # probabilities of each action of input actions 
            # batch at input states batch
            self.actions_prob = tf.gather(params=self.reshaped_policy, 
                                          indices=self.indices,
                                          name="actions_prob")
            self.actions_log_prob = tf.gather(params=self.reshaped_log_policy, 
                                          indices=self.indices,
                                          name="actions_log_prob")
            
            
            self.params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, 
                                            scope=scope)
            

    def predict_batch_policy(self, sess, states):
        return sess.run([self.policy, self.log_policy], {self.states_ph: states})
    
     
        
#tf.reset_default_graph()
#a = PolicyCNN("anan", [84, 84, 4], 5)
#states = np.random.randn(20,84,84,4)
#action = np.arange(20) % 5
#
#loss = tf.reduce_mean(a.actions_prob)
#optimizer = tf.train.GradientDescentOptimizer(0.001)
#train = optimizer.minimize(loss)
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    prob = a.predict_action_probabilities(sess, states, action)
#    probs = a.predict_batch(sess, states)
#    
#    for n in range(2000):
#        sess.run(train, feed_dict={a.actions_ph: action, a.states_ph: states})
#        
#    prob_after = a.predict_action_probabilities(sess, states, action)
#    probs_after = a.predict_batch(sess, states)
#    
#    
#    
#    
    
    
    
    
    
    
    
    
    
    


