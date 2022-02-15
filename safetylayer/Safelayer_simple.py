import torch
from safetylayer.neuralnetwork import MLP
import torch.nn.functional as F

class Safetylayer_sim():
    def __init__(self,
                 env,
                 learning_rate,
                 device="cpu"):
        # Setting the initial environment
        self.env = env
        self.device = device
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.model = MLP(self.input_dim,self.output_dim)
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Setting the initial values of the environment
        self.observation = self.env.reset()
        self.action = self.env.action_space.sample()
        self.c = 0
        self.c_next = 0
        self.num_constraints = 2
        # Setting batch
        self.batch_dict = {
            "obs":[],
            "act":[],
            "c":[],
            "c_next":[]
        }


    def to_device(self):
        return self.model.to(self.device)

    def dict_slice(self,inputdict,start,length):
        keys = inputdict.keys()
        dict_slice = {}
        for i in keys:
            if start+length < len(inputdict[i]):
                dict_slice[i] = inputdict[i][start:start+length]
            else:
                dict_slice[i] = inputdict[i][start:]
        return dict_slice

    def train(self,epoches,batch,batch_size=128):
        self.model.train()
        average_loss = []
        for epoch in range(epoches):
            k = 0
            count = 0
            result_cal = 0
            while k < len(batch["c"]):
                batch_slide = self.dict_slice(batch,k,batch_size)
                results = self.update_batch(batch_slide)
                result_cal = results+result_cal
                k+=batch_size
                count+=1
            result_cal = result_cal/count
            average_loss.append(result_cal)
            print("The epoch is {} and the loss is {}".format(epoch,result_cal))
        return average_loss

    def eval(self):
        self.model.eval()

    def sample_steps(self, num_steps):
        max_length = 1000
        episode_length = 0
        observation = self.env.reset(random_set = True)
        c = self.c
        c = float(c)
        c_next = self.c_next

        for step in range(num_steps):
            action = self.env.action_space.sample()
            observation_next, _, done, _ = self.env.step(action)
            c_next = self.env.constraint_activate(observation_next)
            self.batch_dict["act"].append(action)
            self.batch_dict["obs"].append(observation)
            self.batch_dict["c"].append([c])
            self.batch_dict["c_next"].append([c_next])
            observation = observation_next
            c = c_next
            episode_length = episode_length+1
            # print(self.env.outside_boundary())
            if self.env.outside_boundary() or episode_length>max_length:
                observation = self.env.reset(random_set = True)
                episode_length = 0
        return self.batch_dict

    def calculate_loss(self,batch):
        c = torch.Tensor(batch['c']).to(self.device)
        c_next = torch.Tensor(batch['c_next']).to(self.device)
        action = torch.tensor(batch['act']).to(self.device)
        observations = torch.tensor(batch['obs']).to(self.device)
        gs = torch.zeros(action.shape).to(self.device)
        for i,obs in enumerate(observations):
            obs = obs.float()
            k = self.model.forward(obs)
            # gs.append(k)
            gs[i,:] = k
        c_next_pre = torch.zeros(c_next.shape).to(self.device)
        for i,g in enumerate(gs):
            c_next_pre[i,:] = c[i,:] + torch.mul(g.T,action[i,:])

        losses=torch.mean((c_next - c_next_pre)**2)

        return losses

    def update_batch(self,batch):
        losses = self.calculate_loss(batch)
        self.optimizers.zero_grad()
        losses.backward()
        self.optimizers.step()
        results = losses
        return results

    def get_safe_action(self,obs,act,c):
        self.eval()
        obs = torch.tensor(obs).to(self.device)
        obs = obs.float()
        act = torch.tensor(act).to(self.device)
        c = torch.tensor(c).to(self.device)
        g = self.model(obs)
        # (B,1,A)x(B,A,1) -> (B,1,1) -> (B,)
        numer = torch.mul(g.T,act) + c
        denomin = torch.mul(g.T,g) + 1e-8
        # Equation (5) from Dalal 2018.
        mult = F.relu(numer / denomin)
        correction = mult*g
        action_new = act - correction
        return action_new

