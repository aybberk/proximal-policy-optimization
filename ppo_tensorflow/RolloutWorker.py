from Hyperparameters import GAMMA, LAMBDA, RENDER, ACTIONS
import numpy as np
import matplotlib.pyplot as plt
from Hyperparameters import INPUT_SHAPE, STATE_TYPE, ACTIONS
import tensorflow as tf
import gym
from atari_wrappers import make_atari, wrap_deepmind

def discount(arr, coef):
    '''Comment'''
    length = arr.shape[-1]
    coefs = coef ** np.arange(length)
    return coefs * arr
    
def flatten(arr):
    '''Comment'''
    return arr.reshape(-1, *arr.shape[2:])

class RolloutWorker:
    def __init__(self, agent, env_id, num_envs, timesteps):
        self.agent = agent
        self.num_actions = len(ACTIONS)
        self.num_envs = num_envs
        self.envs = []
        self.timesteps = timesteps
        
        self.states       = np.empty(shape=[num_envs, timesteps + 1, *INPUT_SHAPE], 
                                     dtype=np.float32)
        self.probs        = np.empty(shape=[num_envs, timesteps, self.num_actions],
                                     dtype=np.float32)
        self.log_probs    = np.empty(shape=[num_envs, timesteps, self.num_actions],
                                     dtype=np.float32)
        self.actions      = np.empty(shape=[num_envs, timesteps],
                                     dtype=np.int32)
        self.action_probs = np.empty(shape=[num_envs, timesteps],
                                     dtype=np.float32)
        self.action_log_probs = np.empty(shape=[num_envs, timesteps],
                                     dtype=np.float32)
        self.rewards      = np.empty(shape=[num_envs, timesteps],
                                     dtype=np.float32)
        self.returns      = np.empty(shape=[num_envs, timesteps],
                                   dtype=np.float32)
        self.advantages   = np.empty(shape=[num_envs, timesteps],
                                     dtype=np.float32)
        self.values       = np.empty(shape=[num_envs, timesteps + 1],
                                     dtype=np.float32)
        self.news         = np.empty(shape=[num_envs, timesteps + 1],
                                     dtype=np.bool)
        
        self.last_states     = np.zeros([num_envs, *INPUT_SHAPE])
        self.last_states_new = np.zeros(num_envs, dtype=np.bool)
        
        
        
        for n in range(num_envs):    
            env = make_atari(env_id)
            env = wrap_deepmind(env, frame_stack=True, scale=False)
            self.envs.append(env)
            state = env.reset()
            self.last_states[n] = state
            self.last_states_new[:] = 1
            
        
    def rollout(self, sess):
        self.states[:, 0] = self.last_states[:]
        self.news[:, 0]   = self.last_states_new[:]
        
        for t in range(self.timesteps):
            probs_t, log_probs_t = self.agent.policy_net.predict_batch_policy(sess, self.last_states)
            
            actions_t = np.empty(shape=[self.num_envs], dtype=np.int32)
            rewards_t = np.empty(shape=[self.num_envs], dtype=np.float32)
            
            for n in range(self.num_envs):
                action_n_t = np.random.choice(self.num_actions, p=probs_t[n])
                state_n_t_, reward_n_t, done_n_t_, info = self.envs[n].step(action_n_t)
                self.envs[n].render()
                if done_n_t_:
                    state_n_t_ = self.envs[n].reset()
                self.last_states[n] = state_n_t_[:]
                self.last_states_new[n] = done_n_t_
                actions_t[n] = action_n_t
                rewards_t[n] = reward_n_t
                
#            plt.imshow(self.last_states[0]); plt.show(); print("")
                
            self.probs[:,t] = probs_t
            self.log_probs[:, t] = log_probs_t
            self.action_probs[:,t] = probs_t[np.arange(self.num_envs), actions_t]
            self.action_log_probs[:,t] = log_probs_t[np.arange(self.num_envs), actions_t]
            self.actions[:,t]  = actions_t
            self.rewards[:,t]  = rewards_t
            self.states[:,t+1] = self.last_states
            self.news[:,t+1]   = self.last_states_new
        
        for t in range(self.timesteps + 1):
            states_t = self.states[:, t]
            values_t = self.agent.value_net.predict_batch_values(sess, states_t)
            self.values[:,t] = values_t
         
            
        return self._process()

    def _process(self):
        states = self.states
        actions = self.actions
        probs = self.probs
        log_probs = self.log_probs
        action_probs = self.action_probs
        action_log_probs = self.action_log_probs
        rewards = self.rewards
        news = self.news
        values = self.values
        advantages = self.advantages
        returns = np.empty_like(advantages)
        
        for n in range(self.num_envs):
            lastgaelam = 0
            for t in reversed(range(self.timesteps)):
                nonterminal = 1-news[n,t+1]
                delta = rewards[n,t] + GAMMA * values[n,t+1] * nonterminal - values[n,t]
                advantages[n,t] = lastgaelam = delta + GAMMA * LAMBDA * nonterminal * lastgaelam
            
#            seg["tdlamret"] = seg["adv"] + seg["vpred"]
            returns[n, :] = advantages[n, :] + values[n, :-1]

            
        
               
        return flatten(states[:,:-1]), flatten(probs), flatten(log_probs), \
        flatten(action_probs), flatten(action_log_probs), flatten(actions), \
        flatten(rewards), flatten(returns), flatten(advantages), flatten(values[:,:-1])
        
        
        
        
        