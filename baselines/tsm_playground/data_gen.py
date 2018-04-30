import numpy as np
import gym

from PIL import Image

class Gymmer():
    def __init__(self, env_id):
        self.env = gym.make(env_id)
        self.is_done = True
        self.max_steps = 1024
        self.max_delta = 256
        self.bag_step = 0
        self.frame_bag = np.zeros((self.max_steps,) +self.env.observation_space.shape)
        self.get_pair_count = 0

    def refill_bag(self):
        self.get_pair_count = 0
        self.frame_bag = np.zeros((self.max_steps,) +self.env.observation_space.shape)
        self.env.reset()
        self.bag_step = 0
        last_action = self.env.action_space.sample()
        for s in range(self.max_steps):
            if(np.random.rand(1)[0] > 0.8):
                action = self.env.action_space.sample()
            else:
                action = last_action
            observation, reward, done, info = self.env.step(action)
            last_action = action
            self.frame_bag[s] = observation
            self.bag_step += 1
            if(done):
                break

    def get_pair(self):
        self.get_pair_count += 1
        if(self.bag_step == 0 or self.get_pair_count > self.bag_step/4):
            self.refill_bag()
        left_idx = int(np.random.rand(1)[0]*self.bag_step)
        right_idx = int(np.random.rand(1)[0]*self.max_delta)%self.bag_step
        time_delta = abs(left_idx - right_idx)
        return time_delta, self.frame_bag[left_idx], self.frame_bag[right_idx]

            

    def get_batch(self, size):
        batch = np.zeros((size*2,) +self.env.observation_space.shape)
        times = np.zeros(size)
        for i in range(size):
            td, left, right = self.get_pair()
            batch[2*i] = left
            batch[2*i+1] = right
            times[i] = td


        # Normalize
        times = times/self.max_steps


        return batch, times 

#gm = Gymmer('SpaceInvaders-v0')
#gm.refill_bag()
#batch, times = gm.get_batch(256)
#print(times)
#print(times.mean())
