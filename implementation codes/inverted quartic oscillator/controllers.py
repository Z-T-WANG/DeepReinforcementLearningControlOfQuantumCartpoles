from __main__ import *

#def set_settings(dict):
#    globals().update({k:v for k,v in dict.items() if k not in globals()})
control_time = 1./controls_per_unit_time

def steepest_descent(state, predict_evolution = True, damping = 0.5):
    # the damping factor leaves some portion of the momentum not cancelled by the force
    p_expc = p_expct(state)
    if predict_evolution:
        x3_expc, xpx_expc = np.real(np.conj(state).dot(x**3 * state))*grid_size, xpx_expct(state)
        p_predicted = p_expc -4.*lambda_*x3_expc*control_time-control_time**2 *2.*lambda_*3.*xpx_expc/mass
    else: p_predicted = p_expc
    return -p_predicted/control_time * damping

def LinearQuadratic(state, k):
    # k is the Quadratic loss of the controller
    x_expc, p_expc, x3_expc, xpx_expc = x_expct(state), p_expct(state), np.real(np.conj(state).dot(x**3 * state))*grid_size, xpx_expct(state)
    x_predicted, p_predicted = x_expc+p_expc/mass*control_time-control_time**2 *2.*lambda_*x3_expc/mass, p_expc-4.*lambda_*x3_expc*control_time-control_time**2 *2.*lambda_*3.*xpx_expc/mass
    return -(sqrt(k*mass)*x_predicted+p_predicted)/control_time

def Gaussian_approx(state):
    x_expc, p_expc, x3_expc, xpx_expc = x_expct(state), p_expct(state), np.real(np.conj(state).dot(x**3 * state))*grid_size, xpx_expct(state)
    x_predicted, p_predicted = x_expc, p_expc#+p_expc/mass*control_time #-4.*lambda_*x3_expc*control_time#-control_time**2 *2.*lambda_*x3_expc/mass, #-control_time**2 *2.*lambda_*3.*xpx_expc/mass
    variance = x_2_expct(state)-x_expc**2
    target_p = -sqrt(2*mass*(6*abs(lambda_)*variance+abs(lambda_)*x_predicted**2))*x_predicted
    F = (target_p-p_predicted)/control_time
    #print(p_expc,x_expc, target_p, p_predicted, p_predicted-F/controls_per_unit_time)
    return F
