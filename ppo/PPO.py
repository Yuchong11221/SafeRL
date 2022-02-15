from ppo.layerclass import Actor
from ppo.layerclass import Critic
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal

class PPO():
    def __init__(self,
                 env,
                 lr=0.001,
                 gamma=0.95,
                 clip=0.2,
                 timesteps_per_batch=500,
                 max_timesteps_per_episode=500):
        # Setting environment
        self.env = env
        self.act_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]

        # Setting initial value
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode

        # Setting layer for actor and critic
        self.actor = Actor(self.obs_dim,self.act_dim)
        self.critic = Critic(self.obs_dim,1)

        # Setting optimizer
        self.opt_actor = torch.optim.Adam(self.actor.parameters(),lr=self.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self,total_step,update_per_iter):
        i_now = 0
        for t in range(total_step):
            i_now = i_now + 1
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            for _ in range(update_per_iter):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.opt_actor.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.opt_actor.step()

                # Calculate gradients and perform backward propagation for critic network
                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()

            if i_now % 10 ==0:
                print("iteration is")
                print(i_now)
            # if i_now % 100 == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation.
            obs = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1  # Increment timesteps ran this batch so far
                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # Query the actor network for a mean action
        # print("obs is")
        # print(obs)
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

