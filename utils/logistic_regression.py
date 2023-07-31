import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp, softmax


import time
from scipy.optimize import minimize, LinearConstraint

from utils.nan_operators import nan_logsumexp

# shorter notations
def sum_s(X):
    return np.nansum(X, axis=1, keepdims=True)
def sum_t(X):
    return np.nansum(X, axis=0)
# /!\ sum_s keeps the number of axes intact, sum_t does not

 #%%




def logistic_reg_log_likelihood(uv, r):
    """
    Parameters
    ----------
    uv: np array with shape (2*n_segments,)
    r: np array with shape (n_days, n_segments)

    Returns
    -------
    LL: scalar
    """

    n_days, n_segments = r.shape
    t = np.linspace(0,1,n_days).reshape(-1,1)
    t*uv[np.newaxis,:n_segments] + uv[np.newaxis,n_segments:]

    exponential_arg = t*uv[np.newaxis,:n_segments] + uv[np.newaxis,n_segments:]   # shape: (n_days, n_segments)
    LL = np.nansum(r * (exponential_arg - logsumexp(exponential_arg, axis=1, keepdims=True)))
    return -LL


def compute_gradient(uv, r):
    """
    Parameters
    ----------
    uv: np array with shape (2*n_segments,)
    r: np array with shape (n_days, n_segments)

    Returns
    -------
    gradient: symetric np array with shape (2*n_segments)
    """
    n_days, n_segments = r.shape
    t = np.linspace(0,1,n_days).reshape(-1,1)

    gradient = np.zeros(2*n_segments)
    exponential_arg = t*uv[np.newaxis,:n_segments] + uv[np.newaxis,n_segments:]     # shape: (n_days, n_segments)
    softmax_result =  softmax(exponential_arg, axis=1)                              # shape: (n_days, n_segments)
    gradient[:n_segments] = sum_t(t*r) - sum_t(t*sum_s(r)*softmax_result)
    gradient[n_segments:] = sum_t(  r) - sum_t(  sum_s(r)*softmax_result)
    return -gradient




def compute_hessian(uv, r):
    """
    Parameters
    ----------
    uv: np array with shape (2*n_segments,)
    r: np array with shape (n_days, n_segments) # required because of scipy.optimize

    Returns
    -------
    hessian: symetric np array with shape (2*n_segments, 2*n_segments)
    """
    n_days, n_segments = r.shape
    r = np.random.randn(*r.shape)
    t = np.linspace(0,1,n_days).reshape(-1,1)
    softmax_result = softmax(t*uv[np.newaxis,:n_segments] + uv[np.newaxis,n_segments:], axis=1)  # size: (n_days, n_segments)

    hessian = np.zeros((2*n_segments, 2*n_segments))
    # Case s1 == s2:
        # u with u
    partial_derivatives = sum_t((t**2) * softmax_result * (softmax_result-1))  # shape: (n_segments,)
    hessian[np.arange(n_segments), np.arange(n_segments)] = partial_derivatives
        # u with v:
    partial_derivatives = sum_t(   t   * softmax_result * (softmax_result-1))  # shape: (n_segments,)
    hessian[np.arange(n_segments), np.arange(n_segments)+n_segments] = partial_derivatives
    hessian[np.arange(n_segments)+n_segments, np.arange(n_segments)] = partial_derivatives
        # v with v
    partial_derivatives = sum_t(         softmax_result * (softmax_result-1))  # shape: (n_segments,)
    hessian[np.arange(n_segments)+n_segments, np.arange(n_segments) + n_segments] = partial_derivatives

    # Case s2 != s1:
    for s1 in range(n_segments):
        for s2 in range(s1):
            # u with u
            partial_derivative = sum_t((t[:,0]**2) * softmax_result[:,s1] * softmax_result[:,s2])
                                        # keep in mind both t and sq_sum_s_exp had a shape of (n_days, 1)
            hessian[s1,s2] = partial_derivative
            hessian[s2,s1] = partial_derivative

            # u with v:
            partial_derivative = sum_t(t[:,0] * softmax_result[:,s1] * softmax_result[:,s2])
            hessian[s1, s2+n_segments] = partial_derivative
            hessian[s2+n_segments, s1] = partial_derivative
            hessian[s1+n_segments, s2] = partial_derivative
            hessian[s2, s1+n_segments] = partial_derivative

            # v with v
            partial_derivative = sum_t( softmax_result[:,s1] * softmax_result[:,s2])
            hessian[s1+n_segments,s2+n_segments] = partial_derivative
            hessian[s2+n_segments,s1+n_segments] = partial_derivative


    assert np.allclose(hessian, hessian.T)
    hessian = np.linalg.pinv(np.linalg.pinv(hessian))

    return -hessian





def logistic_regression(r, min_slope=None, init=None):
    """
    This function implements logistic regression where labels are probabilities.
    this function relies on scipy.optimize.minimize, with the option 'trust-constr'. See:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html


    Parameters
    ----------
    r: responsibilities, np array with shape (n_days, n_segments)
        Corresponds to the r_{.,t}^{k,s} in the formula, where t and s are its indices,
        and k is chosen outside of the function.
        /!\ r may contain NaNs
    min_slope: float or None, optional
        provides the minimal slope between two subsequent segments.
    init: couple of numpy arrays (u,v) or None
        The initial values for the parameters.
        If omitted, all parameters start at zero

    Returns
    -------
    u, v: np arrays, each with shape (n_segments,)
    """
    n_segments = r.shape[1]
    if init is None:
        u = np.zeros(n_segments)
        v = np.zeros(n_segments)
        if min_slope is not None:  # initialize within the constraints
            starting_slope = np.abs(min_slope)+1 if min_slope > -np.inf else 1.
            u += 2 * starting_slope * np.arange(n_segments)
            # v -= 2 * starting_slope * np.linspace(1/(2*n_segments), 1-1/(2*n_segments), n_segments)
    else:
        u, v = init
    u = u - u.mean()
    v = v - v.mean()
    initial_uv = np.concatenate([u, v], axis=0).copy()     #   shape (1,2*n_segments)


    u_mean_0 = LinearConstraint(np.concatenate([np.ones(n_segments),  np.zeros(n_segments)], axis=0).reshape(1,-1), lb=0., ub=0., keep_feasible=True)
    v_mean_0 = LinearConstraint(np.concatenate([np.zeros(n_segments), np.ones(n_segments) ], axis=0).reshape(1,-1), lb=0., ub=0., keep_feasible=True)
    constraints = [u_mean_0, v_mean_0]
    if min_slope is not None:
        base_change_diff = -1*np.eye(2*n_segments) + np.diag(np.ones(2*n_segments-1), 1)
        for i in list(range(n_segments-1)):
            ortho_v = base_change_diff[i,:]  # the orthogonal complementary of the subspaces
            const = LinearConstraint(ortho_v.reshape(1,-1), lb=min_slope, ub=np.inf, keep_feasible=True)
            constraints.append(const)

    solution = minimize(fun=lambda uv:logistic_reg_log_likelihood(uv, r), method='trust-constr',
                  jac=lambda uv:compute_gradient(uv, r), hess=lambda uv:compute_hessian(uv, r),
                  x0=initial_uv, constraints=constraints)

    return solution.x[:n_segments], solution.x[n_segments:]












if __name__ == "__main__":

    n_days = 500
    n_segments = 5
    t = np.linspace(0,1,n_days)

    min_slope = 1
    n_segments = 2
    n_pts = 100
    noise_level = 0.1

    noise = noise_level * np.random.randn()

    proba_1 = np.exp(-2*(t - t.mean())/t.std())
    proba_2 = np.exp( 2*(t - t.mean())/t.std())
    sum_probas = proba_1 + proba_2
    proba_1 /= sum_probas
    proba_2 /= sum_probas

    r = np.stack([proba_1, proba_2], axis=-1)  # shape: (n_days, 2)
    r[0,:] = np.nan

    true_u = np.array([    2/t.std(),            -2/t.std()   ])
    true_v = np.array([-2*t.mean()/t.std(), 2*t.mean()/t.std()])
    plt.figure(figsize=(20,8))
    plt.subplot(1,3,1)

    noisy_r = np.clip(r + noise_level * np.random.randn(*r.shape), 0, 1)
    plt.plot(t, noisy_r[:,0], "r.", label='segment 1')
    plt.plot(t, noisy_r[:,1], "b.", label='segment 2')


    u, v = logistic_regression(noisy_r)
    recomputed_probas = softmax(t.reshape(-1,1) * u.reshape(1,-1) + v.reshape(1,-1), axis=1)
    plt.plot(t, recomputed_probas[:,0], c=[1., 0.7, 0.], label="segment 1, reconstructed", linestyle='dotted')
    plt.plot(t, recomputed_probas[:,1], c=[0., 0.7, 1.], label="segment 2, reconstructed", linestyle='dotted')
    plt.title(f"noise level = {noise_level}")
    plt.axis([-0.03, 1.03, -0.15, 1.03]) # leave room for the legend
    plt.legend(ncol=2, loc='lower center')


    plt.show()





#%%
if __name__ == "__main__":


      # more realistic example
    np.random.seed(1)
    n_days = 500
    n_segments = 5
    t = np.linspace(0,1,n_days)
    t = t.reshape(-1,1)

    distances_to_best = {}

    u_gt = np.linspace(-300, 300, n_segments+2)[1:-1]
    u_gt += np.random.uniform(-1/2, 1/2, u_gt.shape) * (u_gt[1:2] - u_gt[0:1])
    v_gt = np.zeros(n_segments)
    v_gt[0] = 0 # left at zero
    segment_borders = np.linspace(0,t.max(), n_segments+1)[1:-1]  # shape: (n_segments-1, )
    segment_borders += np.random.uniform(-1/(2*n_segments), 1/(2*n_segments), size=segment_borders.shape)
    for s in range(1, n_segments):
        v_gt[s] = v_gt[s-1] + segment_borders[s-1] * (u_gt[s-1] - u_gt[s])


    u_gt -= u_gt.mean()
    v_gt -= v_gt.mean()

    exponential = np.exp(t*u_gt + v_gt)
    r = exponential / sum_s(exponential)
    unit_noise = np.random.uniform(-1, 1, size=r.shape)

    r[0,:] = np.nan
    for noise_level in 10**np.arange(-6, 1.1, 0.5):


        r_noisy = r + noise_level*unit_noise
        r_noisy = np.clip(r_noisy, 0, 1)
        noise_description = f"uniform({-noise_level:.0e}, {noise_level:.0e})"

        # r_noisy = np.clip(r, noise_level, 1)
        # noise_description = f"clipping(r, {noise_level:.1e}, 1)"

        # noise_t = noise_level * np.std(u_gt) * np.random.randn(t.shape[0], n_segments)
        # r_noisy = softmax(u_gt * t + v_gt + noise_t, axis=1)
        # noise_description = r"$u_{gt} . t + v_{gt} \leftarrow u_{gt} . t + v_{gt} + \mathcal{N}$" + f"(0, {noise_level:.1e})"

        min_slope_list = [None, -100, 0, 10, 30, 100]

        n_col = 3
        n_rows = 1 + np.ceil(len(min_slope_list)/n_col).astype(int)
        plt.figure(figsize=(10*n_col,13))

        plt.subplot(n_rows, n_col, 1)
        plt.title("noiseless data")
        plt.ylim(-0.05, 1.05)
        for s in range(n_segments):
            plt.plot(r[:,s])



        plt.subplot(n_rows, n_col, 2)
        plt.ylim(-0.05, 1.05)
        plt.title("noisy data \n noise: "+noise_description)
        for s in range(n_segments):
            plt.plot(r_noisy[:,s], "o", markersize=0.5)



        for i, min_slope in enumerate(min_slope_list):
            u, v = logistic_regression(r_noisy, min_slope=min_slope)
            r_recomputed = softmax(t*u + v, axis=1)

            plt.subplot(n_rows, n_col, n_col+i+1)
            plt.ylim(-0.05, 1.05)
            plt.title(f"min_slope={min_slope}")
            for s in range(n_segments):
                plt.plot(r_recomputed[:,s]) #-r_noisy[:,s])


        plt.show()

#%%




# =============================================================================
# Sanity check: Compare the function with gradient descent
# =============================================================================




def project_on_constraints(u, min_slope):



    n_segments = u.shape[0]
    projected_u = u.copy()
    affine_translation = min_slope * np.arange(n_segments) # each boundary is affine subspace
    projected_u -= affine_translation
    base_change_diff = -1*np.eye(n_segments) + np.diag(np.ones(n_segments-1), 1)
    base_change_integral = -np.tri(n_segments).T
    assert np.allclose(base_change_diff @ base_change_integral, np.eye(n_segments))

    for i in range(n_segments-1):
        v = base_change_diff[i,:]  # the orthogonal complementary of the subspace
        u_scalar_v = projected_u @ v
        if u_scalar_v < 0:
            projected_u -=  u_scalar_v * v  /( np.linalg.norm(v)**2)
    return projected_u + affine_translation




you_want_unnecessary_plots = False
if (__name__ == "__main__") and you_want_unnecessary_plots:
    min_slope = 1
    n_segments = 2
    n_pts = 100

    u = np.random.randn(n_pts, n_segments)
    u[:n_segments,:n_segments]= np.eye(n_segments)*2
    u_after_proj = np.zeros_like(u)
    for i in range(n_pts): u_after_proj[i,:] = project_on_constraints(u[i,:], min_slope)
    plt.figure(figsize=(15, 15))
    plt.scatter(u[:,0], u[:,1], label='before', c="b")
    plt.scatter(u_after_proj[:,0], u_after_proj[:,1], label='after\n projection', c="r")
    for i in range(n_pts): plt.plot([u[i,0], u_after_proj[i,0]], [u[i,1], u_after_proj[i,1]], c=[0.9, 0.6, 0.3])

    xmin = u.min()
    xmax = u.max()
    plt.plot([xmin, xmax], [xmin+min_slope, xmax+min_slope], '-.', c='k', label='problem boundary')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

    assert ((u_after_proj[:,1] - u_after_proj[:,0]) >= min_slope-1e-10).all()


#%%

def logistic_regression_gradientdescent(r, min_slope=None, atol=1e-15, rtol=1e-16, init=None, learning_rate=1e-5, momentum_parameter=0.995):
    """
    This function is only here to verify that the scipy implementation
    finds the global maximum (we know gradient descent finds the global
    optimum because We optimize a convex function and the space defined
    by the constraints is convex).
    The only reason why we do not use gradient descent is its computational
    requirement: this function is noticeably slower than the one based on
    scipy.optimize.

    Parameters
    ----------
    r: responsibilities, np array with shape (n_days, n_segments)
        Corresponds to the r_{.,t}^{k,s} in the formula, where t and s are its indices,
        and k is chosen outside of the function.
    min_slope: float or None, optional
        provides the minimal slope between two subsequent segments.
    atol: float, optional
        absolute tolerance for the stopping criterion
    rtol: float, optional
        relative tolerance for the stopping criterion
    init: couple of numpy arrays (u,v) or None
        The initial values for the parameters.
        If omitted, all parameters start at zero
    learning_rate: float
    momentum_parameter: float, defaults to 0.9
    Returns
    -------
    u, v: np arrays, each with shape (n_segments,)
    """

    r_sum = sum_s(r)  # shape: (n_days, 1)
        # corresponds to r_{.,t}^{k,.} in the formula

    n_days, n_segments = r.shape
    t = np.linspace(0,1,n_days)  # .reshape(-1,1)
    t = t[:,np.newaxis]
    if init is None:
        u = np.linspace(-1,  1, n_segments)
        v = np.linspace( 1, -1, n_segments)
        uv = np.concatenate([u, v], axis=0).copy().reshape(1,-1)
    else:
        u, v = init
        uv = np.concatenate([u, v], axis=0).copy().reshape(1,-1)
    #  uv has a shape (1,2*n_segments)

    parameter_history = [uv.copy()]
    previous_likelihood = -np.inf
    likelihood = np.inf
    n_iter = 0
    velocity = np.zeros_like(uv)

    uv[:,:n_segments] -= uv[:,:n_segments].mean()
    uv[:,n_segments:] -= uv[:,n_segments:].mean()
    while np.abs(previous_likelihood - likelihood) >= atol + rtol*np.abs(previous_likelihood) and n_iter < 1e7:
        u, v = uv[:,:n_segments], uv[:,n_segments:]  # both vectors have a shape (1, n_segments)
        exponential = np.exp(t*u + v)                              # shape: (n_days, n_segments)
        previous_likelihood = likelihood
        likelihood = np.nansum(r * np.log(exponential / sum_s(exponential))) #- (1e-2)/(variance + 1e-5)
        sum_s_exp = sum_s(exponential)
        # print(sum_s_exp)
        gradient = np.zeros((1, 2*n_segments))
        gradient[:,:n_segments] = sum_t(r*t) - sum_t(t * r_sum * exponential / sum_s_exp)
        gradient[:,n_segments:] = sum_t(  r) - sum_t(    r_sum * exponential / sum_s_exp)

        gradient[:,:n_segments] -= gradient[:,:n_segments].mean()
        gradient[:,n_segments:] -= gradient[:,n_segments:].mean()

        old_velocity = velocity
        velocity = velocity * momentum_parameter - learning_rate * (-gradient)  # we replace gradient with -gradient because we compute gradient ascent
        uv += -momentum_parameter * old_velocity + (1 + momentum_parameter) * velocity

        uv[:,:n_segments] -= np.nanmean(uv[:,:n_segments])
        uv[:,n_segments:] -= np.nanmean(uv[:,n_segments:])

        if min_slope != None:
            uv[0,:n_segments] = project_on_constraints(uv[0,:n_segments], min_slope)

        parameter_history.append(uv.copy())

        n_iter += 1
    if n_iter == 1e7:
        print('logistic_regression_gradientdescent warning: Convergence was not reached')
    return uv[0,:n_segments], uv[0,n_segments:]






if __name__ == "__main__":

    n_days = 100
    n_segments = 4
    t = np.linspace(0,1,n_days).reshape(-1,1)


    for min_slope in [None, -100, -30, -10, 0, 10, 30, 100]:
        u_gt = np.random.uniform(-1, 1, (1, n_segments)) * n_segments
        v_gt = -u_gt/2
        v_gt += np.random.uniform(-1, 1, (1, n_segments)) * n_segments / 5 # noise
            # these values will likely not be reached because of the constraints we impose to the optimization
            # what matters is tha tthe two algorithms find the same solution
        r = softmax(u_gt * t + v_gt, axis=1)  # shape: (n_days, n_segments)

        r[0,:] = np.nan

        # u_gd, v_gd   = logistic_regression_gradientdescent(r, min_slope=min_slope)
        # u_opt, v_opt = logistic_regression(r, min_slope=min_slope)


        u_gd, v_gd   = logistic_regression_gradientdescent(r, min_slope=min_slope)
        u_opt, v_opt = logistic_regression(r, min_slope=min_slope)



        plt.figure(figsize=(20, 10))
        plt.subplot(1,3,1)
        for s in range(n_segments): plt.plot(r[:,s])
        plt.title(f"min_slope={min_slope}")
        plt.subplot(1,3,2)
        plt.title(f"gradient descent \n min_slope={min_slope}")
        r_gd = softmax(u_gd * t + v_gd, axis=1)
        for s in range(n_segments): plt.plot(r_gd[:,s])
        plt.subplot(1,3,3)
        plt.title(f"scipy.optimize \n  min_slope={min_slope}")
        r_opt = softmax(u_opt * t + v_opt, axis=1)
        for s in range(n_segments): plt.plot(r_opt[:,s])

        print("\n\tmin_slope = ", min_slope)
        print("GD   ", u_gd, v_gd)
        print("scipy", u_opt, v_opt)
        print("GT   ", u_gt.reshape(-1), v_gt.reshape(-1))



