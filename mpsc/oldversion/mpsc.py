import numpy as np
import casadi as cs
import cvxpy as cv
from scipy import linalg
from mpsc.mpc_unit import *
from copy import deepcopy

class MPSC():
    def __init__(self,
                 env,
                 q_lin,
                 r_lin,
                 horizon=10,
                 n_sample=600,
                 tau=0.95,
                 runlength=200,
                 additional_constraint: list = None):

        # Initialize the parameters
        self.env = env
        self.q_lin = q_lin
        self.r_lin = r_lin
        self.horizon = horizon
        self.n_sample = n_sample
        self.tau = tau
        self.runlength = runlength
        # self.model = self.env.sybolic
        # Here I have a question if the dt is 1, since every iteration take 1 step, so the dt is 1
        self.dt = 1
        self.Q = get_cost_weight_matrix(q_lin,self.env.observation_space.shape[0])
        self.R = get_cost_weight_matrix(r_lin,self.env.action_space.shape[0]+1)
        # print(self.Q)
        # print(self.R)

        # Getting constraints from the system
        if additional_constraint is None:
            additional_constraint = []

        # Setting the A,B, and constraints of the system
        self.dfdx_matrix = np.array([[1,1,1,self.dt,0,0],
                                     [1,1,1,0,self.dt,0],
                                     [1,1,1,0,0,self.dt],
                                     [0,0,0,1,0,0],
                                     [0,0,0,0,1,0],
                                     [0,0,0,0,0,1]])

        # Here 0.4 means the weight of the ball, so the a is F/m,which F is the input U, so then 1/m
        self.dfdu_matrix = np.array([[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0],
                                    [1/0.4,0,0],
                                     [0,1/0.4,0],
                                     [0,0,1/0.4]])

        self.constraints = self.env.get_constraint()
        self.constraints = self.constraints.append(additional_constraint)

    def compute_lqr_gain(self):
        """Compute LQR gain by solving the DARE.
        """
        P = linalg.solve_discrete_are(self.dfdx_matrix,
                               self.dfdu_matrix,
                               self.Q,
                               self.R)
        btp = np.dot(self.dfdu_matrix.T, P)
        # print(btp)
        # np.dot(scipy.linalg.inv(np.dot(np.dot(B.T, M), B) + R), (np.dot(np.dot(B.T, M), A)))
        self.lqr_gain = -np.dot(linalg.inv(np.dot(btp, self.dfdu_matrix) + self.R), np.dot(btp,self.dfdx_matrix))
        return self.lqr_gain

    def learn(self):
        # Create set of error residuals.
        w = np.zeros((self.env.observation_space.shape[0], self.n_sample))
        next_true_states = np.zeros((self.env.observation_space.shape[0], self.n_sample))
        next_pred_states = np.zeros((self.env.observation_space.shape[0], self.n_sample))
        actions = np.zeros((self.env.action_space.shape[0]+1, self.n_sample))
        for i in range(self.n_sample):
            init_state, info =self.env.reset()
            u = self.env.action_space.sample()
            actions[:2,i] = u
            x_next_obs, _, _, _ = self.env.step(u)
            # print(x_next_obs.shape)
            # print(self.dfdx_matrix.shape)
            # print(self.dfdu_matrix.shape)
            # print(np.hstack((u,0)))
            # print(u)
            # ????????????????????????linear_dynamics_func????????????????????????????????????AX+BU????????????,????????????x_next_linear???????????????????????????
            x_next_linear = np.dot(self.dfdx_matrix,x_next_obs)+ np.dot(self.dfdu_matrix,actions[:,i])
            next_true_states[:,i] = x_next_obs
            next_pred_states[:,i] = x_next_linear
            w[:,i] = x_next_obs - x_next_linear

        # A_cl = self.dfdx_matrix + np.matmul(self.dfdu_matrix, self.compute_lqr_gain())
        self.learn_action = actions
        self.next_true_states = next_true_states
        self.next_pred_states = next_pred_states
        self.step_optimizer()
        # return next_true_states,next_pred_states

    def step_optimier(self):
        horizons_step = self.horizon
        opti = cs.Opti()
        nx = self.env.observation_space.shape[0]
        nu = self.env.action_space.shape[0]
        z_var = opti.variable(nx,horizons_step+1)
        v_var = opti.variable(nu,horizons_step)

        # Current u_tilda and x
        u_tida = opti.variable(nu,1)
        x = opti.variable(nx,1)

        # Desired input.
        u_L = opti.parameter(nu, 1)

        # implement the mpsc algorithm
        for i in range(horizons_step):
            # ???????????????????????????dynamic function???A???B???w?????????next state
            # Here is the constraint equation from 5-b
            # next_state = self.linear_fun(x=z_var[:i],u=v_var[:i])
            next_state = np.dot(self.dfdx_matrix, z_var[:i]) + np.dot(self.dfdu_matrix, v_var[:i])
            opti.subject_to(z_var[:i+1] == next_state)

            # I think here we need to refer to the paper safe reinforcement learning using robust mpc and get constraint
            # 17-d and 17-e
            # opti.subject_to()
        # Final state constraints (5.d).
        opti.subject_to(z_var[:, -1] == 0)
        # equation 5-f
        opti.subject_to(u_tida == v_var[:,0] + np.dot(self.compute_lqr_gain,(x - z_var[:,0])))

        cost = np.dot((u_L - u_tida).T,(u_L - u_tida))
        opti.minimize(cost)
        # Create solver (IPOPT solver as of this version).
        opts = {"ipopt.print_level": 4,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 50,
                "print_time": 1}
        opti.solver('ipopt', opts)

        self.opti_dict = {
            "opti": opti,
            "z_var": z_var,
            "v_var": v_var,
            "u_tida": u_tida,
            "u_L": u_L,
            "x": x,
            "cost": cost
        }


    def solve_optimization(self,
                           obs,
                           uncertified_input
                           ):
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        z_var = opti_dict["z_var"]
        v_var = opti_dict["v_var"]
        u_tilde = opti_dict["u_tilde"]
        u_L = opti_dict["u_L"]
        x = opti_dict["x"]
        cost = opti_dict["cost"]
        opti.set_value(x, obs)
        opti.set_value(u_L, uncertified_input)
        # Initial guess for optimization problem.
        if (self.warmstart and
                self.z_prev is not None and
                self.v_prev is not None and
                self.u_tilde_prev is not None):
            # Shift previous solutions by 1 step.
            z_guess = deepcopy(self.x_prev)
            v_guess = deepcopy(self.u_prev)
            z_guess[:, :-1] = z_guess[:, 1:]
            v_guess[:-1] = v_guess[1:]
            opti.set_initial(z_var, z_guess)
            opti.set_initial(v_var, v_guess)
            opti.set_initial(u_tilde, deepcopy(self.u_tilde_prev))
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val, u_tilde_val = sol.value(z_var), sol.value(v_var), sol.value(u_tilde)
            self.z_prev = x_val
            self.v_prev = u_val
            self.u_tilde_prev = u_tilde_val
            # Take the first one from solved action sequence.
            if u_val.ndim > 1:
                action = u_tilde_val
            else:
                action = u_tilde_val
            self.prev_action = u_tilde_val
            feasible = True
        except RuntimeError:
            feasible = False
            action = None
        return action, feasible


    def certify_action(self,
                       obs,
                       u_L
                       ):

        action, feasible = self.solve_optimization(obs, u_L)
        self.results_dict['feasible'].append(feasible)
        if feasible:
            self.kinf = 0
            self.results_dict['kinf'].append(self.kinf)
            return action
        else:
            self.kinf += 1
            self.results_dict['kinf'].append(self.kinf)
            if (self.kinf <= self.horizon - 1 and
                    self.z_prev is not None and
                    self.v_prev is not None):
                action = self.v_prev[self.kinf] + np.matmul(self.lqr_gain,(obs - self.z_prev[:, self.kinf, None]))
                return action
            else:
                action =  np.matmul(self.lqr_gain,obs)
                return action

    def select_action(self,
                      obs
                      ):
        if self.rl_controller is not None:
            with torch.no_grad():
                u_L, v, logp = self.rl_controller.agent.ac.step(torch.FloatTensor(obs).to(self.rl_controller.device))
        else:
            u_L = 2*np.sin(0.01*np.pi*self.time_step) + 0.5*np.sin(0.12*np.pi*self.time_step)
        self.results_dict['learning_actions'].append(u_L)
        action = self.certify_action(obs, u_L)
        action_diff = np.linalg.norm(u_L - action)
        self.results_dict['corrections'].append(action_diff)
        return action, u_L



    def run(self):
        self.setup_results_dict()
        obs, _ = self.env.reset()
        self.setup_results_dict['obs'].append(obs)
        self.kinf = self.horizon -1
        self.timestep = 0
        for i in range(self.runlength):
            act, u_L = self.select_action(obs)
            obs,reward,done,_ = self.env.step(act)
            self.results_dict['obs'].append(obs)
            self.results_dict['actions'].append(action)
            self.time_step += 1

            return self.results_dict

    def setup_results_dict(self):
        self.results_dict = {}
        self.results_dict['obs'] = []
        self.results_dict['actions'] = []
        self.results_dict['cost'] = []
        self.results_dict['learning_actions'] = []
        self.results_dict['corrections'] = [0.0]
        self.results_dict['feasible'] = []
        self.results_dict['kinf'] = []

    def close(self):
        self.env.close()
