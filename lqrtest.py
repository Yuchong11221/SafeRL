import numpy as np
from scipy import linalg

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

dfdx_matrix = np.array([[1,1],
                        [0,1]]
                       )

Q_matrix = np.eye(2)*0.2
print(Q_matrix.shape)

R_matrix = np.ones((1,1))*0.1
# print(R_matrix.shape)
# R_matrix = 0.2

dfdu_matrix = np.array([[0],
                        [0.25]])
# print(dfdu_matrix*R_matrix)

k = compute_lqr_gain(dfdx_matrix,dfdu_matrix,Q_matrix,R_matrix)
print(k)