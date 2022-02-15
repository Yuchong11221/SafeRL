import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetylayer.neuralnetwork import MLP



class SafetyLayerComplex:
    def __init__(self,
                env,
                num_constraints=2,
                lr=0.001,
                device="cpu"
                 ):
        self.device = device
        self.num_constraints = num_constraints
        # Seperate model per constraint.
        self.env = env
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.shape[0]

        # default 1 layer
        self.constraint_models = nn.ModuleList([
            MLP(input_dim, output_dim)
            for _ in range(self.num_constraints)
        ])

        # print(self.constraint_models)

        # Optimizers.
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=lr)
            for model in self.constraint_models
        ]

        self.batch_dict = {
            "obs": [],
            "act": [],
            "c": [],
            "c_next": []
        }

    def to(self,
           device
           ):
        """Puts agent to device.
        """
        self.constraint_models.to(device)

    def train(self,epoches,batch):
        """Sets training mode.
        """
        self.constraint_models.train()
        for epoch in range(epoches):
            result = self.update(batch)
            # print(result)

    def eval(self):
        """Sets evaluation mode.
        """
        self.constraint_models.eval()

    def sample_steps(self, num_steps):
        max_length = 1000
        episode_length = 0
        observation = self.env.reset()
        c = self.env.get_constraint_value(observation)
        c_next = np.array([0,0])

        for step in range(num_steps):
            action = self.env.action_space.sample()
            observation_next, _, done, _ = self.env.step(action)
            c_next = self.env.get_constraint_value(observation_next)
            self.batch_dict["act"].append(action)
            self.batch_dict["obs"].append(observation)
            self.batch_dict["c"].append([c])
            self.batch_dict["c_next"].append([c_next])
            observation = observation_next
            c = c_next
            episode_length += 1
            if self.env.outside_boundary() or episode_length>max_length:
                observation = self.env.reset(choose=True)
                print(observation)
                episode_length = 0
        return self.batch_dict

    def compute_loss(self,
                     batch
                     ):
        obs, act = batch["obs"], batch["act"]
        c, c_next = batch["c"], batch["c_next"]
        obs = torch.tensor(obs).to(self.device)
        act = torch.tensor(act).to(self.device)
        c = torch.tensor(c).to(self.device)
        c_next = torch.tensor(c_next).to(self.device)
        obs = obs.float()

        gs = [model(obs) for model in self.constraint_models]
        # print("gs shape is")
        # print(len(gs))

        # print(c[:,1])
        # for i, g in enumerate(gs):
        #     k = torch.bmm(g.view(g.shape[0], 1, -1),
        #               act.view(act.shape[0], -1, 1)).view(-1)
        #     print("the i th is{}".format(i))
        #     print(k)
        # Each is (N,1,A) x (N,A,1) -> (N,), so [(N,)]_{n_constriants}
        c = c.squeeze(1)
        c_next = c.squeeze(1)
        c_next_pred = [
            c[:,i]+ torch.bmm(g.view(g.shape[0], 1, -1),
                                act.view(act.shape[0], -1, 1)).view(-1)
            for i, g in enumerate(gs)
        ]
        losses = [
            torch.mean((c_next[:, i] - c_next_pred[i]) ** 2)
            for i in range(self.num_constraints)
        ]
        return losses

    def update(self, batch):
        losses = self.compute_loss(batch)
        for loss, opt in zip(losses, self.optimizers):
            opt.zero_grad()
            loss.backward()
            opt.step()
        results = {
            "constraint_{}_loss".format(i): loss.item()
            for i, loss in enumerate(losses)
        }
        return results

    def get_safe_action(self,
                        obs,
                        act,
                        c
                        ):
        self.eval()
        # [(B,A)]_C
        obs = torch.tensor(obs).to(self.device)
        obs = obs.float()
        act = torch.tensor(act).to(self.device)
        c = torch.tensor(c).to(self.device)
        g = [model(obs) for model in self.constraint_models]
        # Find the lagrange multipliers [(B,)]_C
        multipliers = []
        # 现在的问题是输出的action维度有毛病，所以跑不了。看看维度的问题
        for i in range(len(g)):
            g_i = g[i]  # (B,A)
            c_i = c[i]  # (B,)
            # print(torch.mul(g_i,act))
            # (B,1,A)x(B,A,1) -> (B,1,1) -> (B,)
            numer = torch.mul(g_i.T,act) + c_i
            denomin = torch.mul(g_i.T,g_i).view(-1) + 1e-8
            # Equation (5) from Dalal 2018.
            mult = F.relu(numer / denomin)  # (B,)
            multipliers.append(mult)
        multipliers = torch.stack(multipliers, -1)  # (B,C)
        # Calculate correction, equation (6) from Dalal 2018.
        max_mult, max_idx = torch.topk(multipliers, 1, dim=-1)  # (B,1)
        max_idx = max_idx.view(-1).tolist()  # []_B
        # [(A,)]_B -> (B,A)
        max_g = torch.stack([g[max_i][i] for i, max_i in enumerate(max_idx)])
        # (B,1) x (B,A) -> (B,A)
        correction = torch.mul(max_mult.T, max_g)
        action_new = act - correction
        return action_new