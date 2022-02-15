from scipy import linalg
import casadi as cs
import numpy as np

def compute_lqr_gain(dfdx_matrix,dfdu_matrix,Q,R):
    """Compute LQR gain by solving the DARE.
    """
    P = linalg.solve_discrete_are(dfdx_matrix,
                                  dfdu_matrix,
                                  Q,
                                  R)
    # print(self.R.shape)
    # print(self.dfdu_matrix.shape)
    btp = np.dot(dfdu_matrix.T, P)
    lqr_gain = -np.dot(linalg.inv(np.dot(btp, dfdu_matrix) + R), np.dot(btp, dfdx_matrix))
    return lqr_gain

def step_optimizer(dfdx_matrix,dfdu_matrix,Q,R):
    horizons_step = 30
    opti = cs.Opti()
    nx = 2
    nu = 1
    z_var = opti.variable(nx,horizons_step+1)
    v_var = opti.variable(nu,horizons_step)
    # print(z_var.shape)
    # print(v_var.shape)

    # Current u_tilda and x
    u_tida = opti.variable(nu,1)
    x = opti.parameter(nx,1)
    # x = opti.variable(nx,1)
    # Desired input
    u_L = opti.parameter(nu, 1)
    # u_L = opti.variable(nu,1)

    # implement the mpsc algorithm
    for i in range(horizons_step):
        # 在这里我们需要确定dynamic function的A，B，w来构建next state
        # Here is the constraint equation from 5-b
        # next_state = self.linear_fun(x=z_var[:i],u=v_var[:i])
        next_state = dfdx_matrix @ z_var[:,i] + dfdu_matrix @ v_var[:,i]
        opti.subject_to(z_var[:,i+1] == next_state)
        # I think here we need to refer to the paper safe reinforcement learning using robust mpc and get constraint
        # Constraints (currently only handles a single constraint for state and input).
        # 这一处需要根据系统情况进行设计，也可以手动输入，回头估计要进行手动输入操作
        # state_constraints = constraints
        # 5-d and 5-e
        # State Constraints
        # opti.subject_to(cs.SX.fabs(z_var[0,i] -7.5) < 0)
        opti.subject_to((z_var[0,i]-7.5) < 0.0)
        opti.subject_to((-7.5-z_var[0,i])<0.0)
    # Final state constraints (5.d),z需要在0点附近，这也是MPSC最后希望看的
    opti.subject_to((z_var[:, -1])**2<0.1)
    # Final state constraints (5.e).
    opti.subject_to((x-z_var[:,0])**2<0.1)
    # equation 5-f
    # print("lqr_gain is")

    opti.subject_to(u_tida == v_var[:,0] + compute_lqr_gain(dfdx_matrix,dfdu_matrix,Q,R)@(x-z_var[:,0]))

    cost = (u_L - u_tida).T@(u_L - u_tida)
    opti.minimize(cost)
    # Create solver (IPOPT solver as of this version).
    opts = {"ipopt.print_level": 5,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 50,
            "print_time": 1}
    opti.solver('ipopt',opts)

    opti_dict = {
        "opti": opti,
        "z_var": z_var,
        "v_var": v_var,
        "u_tida": u_tida,
        "u_L": u_L,
        "x": x,
        "cost": cost
    }
    return opti_dict

def solve_optimization(obs,uncertified_input):
    opti_dict = step_optimizer(dfdx_matrix,dfdu_matrix,Q_matrix,R_matrix)
    # opti_dict = opti_dict
    opti = opti_dict["opti"]
    z_var = opti_dict["z_var"]
    print(z_var.shape)
    v_var = opti_dict["v_var"]
    print(v_var.shape)
    u_tida = opti_dict["u_tida"]

    u_L = opti_dict["u_L"]
    x = opti_dict["x"]
    cost = opti_dict["cost"]
    # print("cost ist")
    # print(cost)
    opti.set_value(x, obs)
    opti.set_value(u_L, uncertified_input)
    # Solve the optimization problem.
    # try:
    sol = opti.solve()
    x_val =opti.value(z_var)
    u_val = opti.value(v_var)
    u_tilde_val = opti.value(u_tida)
    print("x_val is")
    print(x_val)
    # print("x_val is")
    # print(x_val)
    z_prev = x_val
    v_prev = u_val
    u_tilde_prev = u_tilde_val
    print("u tilde is")
    print(u_tilde_prev)
    # Take the first one from solved action sequence.
    if u_val.ndim > 1:
        action = u_tilde_val
    else:
        action = u_tilde_val

    # _action = u_tilde_val
    feasible = True

    # except RuntimeError:
    #     feasible = False
    #     action = None
    # print("feasible are")
    # print(feasible)
    return action,feasible

# Setting the A,B, and constraints of the system
dt = 0.01
dfdx_matrix = np.array([[1,dt],[0,1]])
dfdu_matrix = np.array([[0],
                        [dt]])
Q_matrix = np.eye(2)
R_matrix = np.ones((1,1))*0.1
# Here 0.4 means the weight of the ball, so the a is F/m,which F is the input U, so then 1/m

# 测试第二个
dict = step_optimizer(dfdx_matrix,dfdu_matrix,Q_matrix,R_matrix)
print(dict)

act1, fea1 = solve_optimization([-3,0],-1)
print("action1 is")
print(act1)
print(fea1)