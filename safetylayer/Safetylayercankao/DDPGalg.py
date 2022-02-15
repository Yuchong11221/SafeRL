import torch
import torch.nn as nn
import torch.nn.functional as F
from replaybuffer import replay_memory

device = "cuda"
class Critic(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc1.weight.data.normal_(0,0.2)
        self.fc2 = nn.Linear(64 + act_dim, 64)
        self.fc2.weight.data.normal_(0,0.2)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.normal_(0,0.2)

    def forward(self, state, action):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(torch.cat((x, action), 1))
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, max_a,gamma=0.9,tau=0.2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64, act_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.max_a = max_a
        self.gamma = gamma
        self.tau = tau

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x) * self.max_a

class DDPG():
    def __init__(self,n_state,n_action,max_action,lr,memory_size=3200):
        self.n_state = n_state
        self.n_action = n_action
        self.max_action = max_action
        self.lr = lr

        # Inply the Actor and Critic
        self.actor = Actor(self.n_state,self.n_action,self.max_action).to(device)
        self.target_actor = Actor(self.n_state,self.n_action,self.max_action).to(device)
        self.critic = Critic(self.n_state,self.n_action).to(device)
        self.target_critic = Critic(self.n_state,self.n_action).to(device)
        self.memory = replay_memory(memory_size)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),self.lr)

    def update_actor(self,batch):
        batch_state = torch.Tensor(batch['observation'].tolist()).to(device)
        action = self.actor(batch_state)
        loss = torch.mean(self.critic(batch_state,action))
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def update_critic(self,batch):
        batch_reward = torch.Tensor(batch['reward'].tolist()).to(device)
        batch_action = torch.Tensor(batch['action'].tolist()).to(device)
        batch_next_state = torch.Tensor(batch['observation_next'].tolist()).to(device)
        batch_state = torch.Tensor(batch['observation'].tolist()).to(device)

        action_next = self.target_actor(batch_next_state)
        # q_target = self.target_critic(batch_next_state,action_next)
        q_state = self.critic(batch_state,batch_action)
        q_target_batch = self.critic(batch_next_state,action_next)
        for i in range(batch_reward.shape[0]):
            q_target_batch[i] = batch_reward[i] + self.gamma*self.critic(batch_next_state,action_next)

        td_error = q_state - q_target_batch.detach()
        loss = (td_error ** 2).mean()
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

    def network_update(self):
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)








