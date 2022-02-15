import numpy as np
import casadi as cs
import cvxpy as cv
from scipy import linalg
from mpsc.mpc_unit import *
from copy import deepcopy

# 在第二个版本中修改了dfdx与dfdu函数及相关计算方法，我觉得应该进一步修改envsimple
class MPSC_SIM():
    def __init__(self,
                 env,
                 q_lin,
                 r_lin,
                 maxforce,
                 horizon=70,
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
        self.maxforce = maxforce
        # self.model = self.env.sybolic
        self.dt = 0.01
        self.Q = get_cost_weight_matrix(q_lin,self.env.observation_space.shape[0])
        self.R = get_cost_weight_matrix(r_lin,self.env.action_space.shape[0])

        # Getting constraints from the system
        if additional_constraint is None:
            additional_constraint = []

        # Setting the A,B, and constraints of the system
        # self.dfdx_matrix,self.dfdu_matrix, self.linear_dy = self.set_linear_dynamics()
        self.dfdx_matrix, self.dfdu_matrix = self.set_linear_dynamics()
        # self.dfdx_matrix = np.array([[1,self.dt],
        #                              [0,1]])

        # Here 0.4 means the weight of the ball, so the a is F/m,which F is the input U, so then 1/m
        # self.dfdu_matrix = np.array([[0],
        #                             [1/0.4*self.dt]])

        self.constraints = self.env.get_constraint() + additional_constraint
        # self.constraints = self.constraints.append(additional_constraint)
        self.z_prev = None
        self.v_prev = None

    def set_linear_dynamics(self):
        """Compute the linear dynamics
        """
        # Original version, used in shooting.
        nx = self.env.observation_space.shape[0]
        nu = self.env.action_space.shape[0]
        # 权宜之计，在之后再调
        # dfdxdfdu = self.model.df_func(x=self.X_LIN, u=self.U_LIN)
        # dfdx = np.array([[0,0,0,1,0,0],
        #                [0,0,0,0,1,0],
        #                [0,0,0,0,0,1],
        #                [0,0,0,0,0,0],
        #                [0,0,0,0,0,0],
        #                [0,0,0,0,0,0]])
        dfdx = np.array([[0, 1], [0, 0]])
        # dfdu = np.array([[0,0,0],
        #                [0,0,0],
        #                [0,0,0],
        #                [1/4,0,0],
        #                [0,1/4,0],
        #                [0,0,1/4]])
        dfdu = np.array([[0], [1 / 4]])
        delta_x = cs.MX.sym('delta_x', nx, 1)
        delta_u = cs.MX.sym('delta_u', nu, 1)
        x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
        linear_dynamics_func = cs.integrator(
            'linear_discrete_dynamics', 'cvodes',
            {
                'x': delta_x,
                'p': delta_u,
                'ode': x_dot_lin_vec
            }, {'tf': self.dt}
        )
        discrete_dfdx, discrete_dfdu = discretize_linear_system(dfdx, dfdu, self.dt)
        return discrete_dfdx, discrete_dfdu#, linear_dynamics_func

    def compute_lqr_gain(self):
        """Compute LQR gain by solving the DARE.
        """
        P = linalg.solve_discrete_are(self.dfdx_matrix,
                               self.dfdu_matrix,
                               self.Q,
                               self.R)
        btp = np.dot(self.dfdu_matrix.T, P)
        self.lqr_gain = -np.dot(linalg.inv(np.dot(btp, self.dfdu_matrix) + self.R), np.dot(btp,self.dfdx_matrix))
        return self.lqr_gain

    # def learn(self):
    #     # Create set of error residuals.
    #     w = np.zeros((self.env.observation_space.shape[0], self.n_sample))
    #     next_true_states = np.zeros((self.env.observation_space.shape[0], self.n_sample))
    #     next_pred_states = np.zeros((self.env.observation_space.shape[0], self.n_sample))
    #     actions = np.zeros((3, self.n_sample))
    #     self.env.reset()
    #     for i in range(self.n_sample):
    #         # 这次我决定只用一个恒定的负方向走
    #         u = -0.2
    #         actions[0,i] = u
    #         x_next_obs, _, _, _ = self.env.step(u)
    #
    #         # 有点问题，这里的linear_dynamics_func有什么用？我在这里直接用AX+BU进行代替,而且在此x_next_linear为根据自己所需改的
    #         z_next_linear = self.dfdx_matrix@x_next_obs + self.dfdu_matrix@actions[:,i]
    #         # print(z_next_linear.shape)
    #         next_true_states[:,i] = x_next_obs
    #         # next_pred_states[:,i] = z_next_linear
    #         # w[:,i] = x_next_obs - z_next_linear
    #
    #     # A_cl = self.dfdx_matrix + np.matmul(self.dfdu_matrix, self.compute_lqr_gain())
    #     # self.learn_action = actions
    #     # self.next_true_states = next_true_states
    #     # self.next_pred_states = next_pred_states
    #     self.step_optimizer()
    #     # return next_true_states,next_pred_states

    def step_optimizer(self):
        horizons_step = self.horizon
        opti = cs.Opti()
        nx = self.env.observation_space.shape[0]
        nu = self.env.action_space.shape[0]
        z_var = opti.variable(nx,horizons_step+1)
        v_var = opti.variable(nu,horizons_step)

        # Current u_tilda and x
        u_tida = opti.variable(nu,1)
        x = opti.parameter(nx,1)
        # Desired input.
        u_L = opti.parameter(nu, 1)

        # implement the mpsc algorithm
        for i in range(horizons_step):
            # 在这里我们需要确定dynamic function的A，B，w来构建next state
            # Here is the constraint equation from 5-b
            next_state = self.dfdx_matrix @ z_var[:,i] + self.dfdu_matrix @ (v_var[:,i]*self.maxforce)
            # print(next_state.shape)
            opti.subject_to(z_var[:,i+1] == next_state)
            # I think here we need to refer to the paper safe reinforcement learning using robust mpc and get constraint
            # Constraints (currently only handles a single constraint for state and input).
            # 这一处需要根据系统情况进行设计，也可以手动输入，回头估计要进行手动输入操作
            state_constraints = self.constraints
            # 5-d and 5-e
            # State Constraints
            for constraint in state_constraints:
                opti.subject_to(constraint(z_var[:, i]) < 0)
            # Input Constraints
            opti.subject_to(v_var[:,i]*self.maxforce<30)
            opti.subject_to(-30<v_var[:,i]*self.maxforce)
        # Final state constraints (5.d),z需要在0点附近，这也是MPSC最后希望看的
        opti.subject_to(z_var[0, -1]<0.1)
        opti.subject_to(z_var[0,-1]>-0.1)
        # Final state constraints (5.e).
        opti.subject_to((x-z_var[:,0])**2<0.1)
        # equation 5-f
        opti.subject_to(u_tida == v_var[:,0] + self.compute_lqr_gain()@(x-z_var[:,0]))

        cost = (u_L - u_tida).T@(u_L - u_tida)
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
        self.step_optimizer()
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        z_var = opti_dict["z_var"]
        v_var = opti_dict["v_var"]
        u_tilde = opti_dict["u_tida"]
        u_L = opti_dict["u_L"]
        x = opti_dict["x"]
        cost = opti_dict["cost"]
        opti.set_value(x, obs)
        opti.set_value(u_L, uncertified_input)
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
                    self.v_prev is not None and
                    self.u_tilde_prev is not None
            ):
                action = self.v_prev[self.kinf]+ self.lqr_gain@(obs - self.z_prev[:, self.kinf])
            else:
                action = self.lqr_gain@obs
            return action

    def select_action(self,
                      obs,
                      u_L
                      ):
        self.results_dict['learning_actions'].append(u_L)
        action = self.certify_action(obs, u_L)
        action_diff = np.linalg.norm(u_L - action)
        self.results_dict['corrections'].append(action_diff)
        return action, u_L

    def run(self,models):
        self.setup_results_dict()
        obs = self.env.reset()
        ref = obs
        print(obs.shape)
        obs = obs-ref
        print(obs.shape)
        self.results_dict['obs'].append(obs)
        self.kinf = self.horizon - 1
        self.time_step = 0
        for i in range(self.runlength):
            act = models(obs)
            act = act.detach().numpy()
            action, u_L = self.select_action(obs,act)
            obs, _, _, ref = self.env.step(action)
            obs = obs - ref
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
