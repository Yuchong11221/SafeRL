import matplotlib.pyplot as plt
import numpy as np
import torch
from env.Envsimple_third import Env_simple
from env.EnvV3 import Env_version3
from safetylayer.Safelayer_simple import  Safetylayer_sim
# from safetylayer.safetycomplex import SafetyLayerComplex
from safetylayer.neuralnetwork import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

# # Here is the part for simple environment and I achieve it by setting higher batch size and higher training epoches
env = Env_simple()
env.reset()
obs = env.reset()
batch = 6000
# # 这个部分属于Safety Layer方向
S_model = Safetylayer_sim(env,device=device,learning_rate=0.01)
S_model.to_device()
batch_s = S_model.sample_steps(batch)

# # update = S_model.update_batch(batch_s)
# # print(update)
loss = S_model.train(10,batch_s)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(loss)
plt.show()

obs = env.reset()
# k = 0
# c = env.constraint_activate(obs)
# act = -0.1
# act_safe = S_model.get_safe_action(obs,act,c)
# print(act_safe)
for i in range(1000):
    env.render()
    act = -0.2
    c = env.constraint_activate(obs)
    act_safe = S_model.get_safe_action(obs,act,c)
    print(act_safe)
    obs,reward,done,_ = env.step(act_safe)
    if env.outside_boundary():
        print("the agent has violate the constraint")
        print(obs)
#         k = k+1
#         print("the number of violation are")
#         print(k)


# Here is the part for 2 dimension environment
# env = Env_version3()
# obs=env.reset(random_reset=False)
# for i in range(2000):
#     env.render()
#     act = [-0.5,0]
#     obs,reward,done,_ = env.step(act)
#     if done:
#         obs = env.reset(random_reset=True)
# # print("obs is")
# # print(obs)
# batch_size = 20000
# S_model = SafetyLayerComplex(env,device=device)
# S_model.to(device=device)
# batch_sc = S_model.sample_steps(batch_size)
# S_model.train(30,batch_sc)
# # print(result)
# obs=env.reset(choose=False)
# # Now the training is still bad
# for i in range(1000):
#     env.render()
#     act = np.array([0.01,0])
#     # print("act shape")
#     # print(act.shape)
#     c = env.get_constraint_value(obs)
#     act_safe = S_model.get_safe_action(obs,act,c)
#     # print("act_safe is")
#     # print(act_safe.shape)
#     act_safe = act_safe.squeeze(0)
#     print(act_safe[0])
#     obs,reward,done,_ = env.step(act_safe)
#     # obs,reward,done,_ = env.step(act)
#     if done:
#         print("the agent has violate the constraint")
#         print(obs)
#         k = k+1
#         print("the number of violation are")
#         print(k)
