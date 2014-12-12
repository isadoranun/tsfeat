import numpy as np
from scipy.optimize import minimize


def CAR_Lik(parameters, t, x, error_vars):

    sigma = parameters[0]
    tau = parameters[1]
    #b = parameters[1] #comment it to do 2 pars estimation
    #tau = params(1,1);
    #sigma = sqrt(2*var(x)/tau);

    b = np.mean(x) / tau
    epsilon = 1e-300
    cte_neg = -np.infty
    num_datos = np.size(x)

    Omega = []
    x_hat = []
    a = []
    x_ast = []

    Omega.append((tau * (sigma ** 2)) / 2.)
    x_hat.append(0.)
    a.append(0.)
    x_ast.append(x[0] - b * tau)

    loglik = 0.

    for i in range(1, num_datos):

        a_new = np.exp(-(t[i] - t[i - 1]) / tau)
        x_ast.append(x[i] - b * tau)
        x_hat.append(
            a_new * x_hat[i - 1] +
            (a_new * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) *
            (x_ast[i - 1] - x_hat[i - 1])
        )

        Omega.append(
            Omega[0] * (1 - (a_new ** 2)) +
            ((a_new ** 2)) * Omega[i - 1] *
            (1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])))
        )

        loglik_inter = np.log(
            ((2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5) *
            (np.exp(-0.5 * (((x_hat[i] - x_ast[i]) ** 2) /
             (Omega[i] + error_vars[i]))) + epsilon)
        )

        loglik = loglik + loglik_inter

        if(loglik <= cte_neg):
            print('CAR lik se fue a inf')
            return None

    #the minus one is to perfor maximization using the minimize function
    return -loglik

data = np.random.uniform(-5, -3, 1000)
error = np.random.uniform(0.000001, 1, 1000)
mjd = np.random.uniform(40000, 50000, 1000)

LC = [mjd, data, error]


def CAR_features(LC):
    x0 = [10, 1]
    bnds = ((0, 100), (0, 100))
    res = minimize(CAR_Lik, x0, args=(LC[:, 0], LC[:, 1], LC[:, 2]),
                   method='nelder-mead', bounds = bnds)
    sigma = res.x[0]
    tau = res.x[1]
    return sigma, tau

CAR_features(LC)
