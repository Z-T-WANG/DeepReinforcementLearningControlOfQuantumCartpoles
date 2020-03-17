import numpy as np
from math import *
from scipy.sparse import csr_matrix as csr

def set_global(x_max = 8.5, x_n_ = 171, lambda_ = pi/25., mass = 1./pi):
    global grid_size, x, x_csr
    x_n = x_n_-1
    grid_size = x_max*2/x_n
    x = np.linspace(-x_max,x_max,x_n+1, dtype=np.float64)
    x_csr = csr(np.diag(x))

    global delta_x, delta_2_x
    delta_x = np.zeros((x_n+1, x_n+1), dtype=np.float64)
    di1, di2 = np.diag_indices(x_n+1)
    delta_x[(di1[:-1],di2[1:])]=672/840; delta_x[(di1[1:],di2[:-1])]=-672/840
    delta_x[(di1[:-2],di2[2:])]=-168/840; delta_x[(di1[2:],di2[:-2])]=168/840
    delta_x[(di1[:-3],di2[3:])]=32/840; delta_x[(di1[3:],di2[:-3])]=-32/840
    delta_x[(di1[:-4],di2[4:])]=-3/840; delta_x[(di1[4:],di2[:-4])]=3/840
    delta_x /= grid_size
    delta_x = csr(delta_x)
    delta_x.prune()
    
    delta_2_x = np.zeros((x_n+1, x_n+1), dtype=np.float64)
    delta_2_x[(di1,di2)]=-14350/5040
    delta_2_x[(di1[:-1],di2[1:])]=8064/5040; delta_2_x[(di1[1:],di2[:-1])]=8064/5040
    delta_2_x[(di1[:-2],di2[2:])]=-1008/5040; delta_2_x[(di1[2:],di2[:-2])]=-1008/5040
    delta_2_x[(di1[:-3],di2[3:])]=128/5040; delta_2_x[(di1[3:],di2[:-3])]=128/5040
    delta_2_x[(di1[:-4],di2[4:])]=-9/5040; delta_2_x[(di1[4:],di2[:-4])]=-9/5040
    # the following lower order estimations around the border of the simulation space will make the Hamiltonian non-Hermitian, so we avoid them
    #delta_2_x[0,:5]=np.array([-2.,1.,0.,0.,0.])/1.
    #delta_2_x[1,:6]=np.array([1.,-2.,1.,0.,0.,0.])/1.
    #delta_2_x[2,:7]=np.array([-1.,16.,-30.,16.,-1.,0.,0.])/12.
    #delta_2_x[3,:8]=np.array([2.,-27.,270.,-490.,270.,-27.,2.,0.])/180.
    #delta_2_x[-1,-5:]=np.array([-2.,1.,0.,0.,0.])[::-1]/1.
    #delta_2_x[-2,-6:]=np.array([1.,-2.,1.,0.,0.,0.])[::-1]/1.
    #delta_2_x[-3,-7:]=np.array([-1.,16.,-30.,16.,-1.,0.,0.])[::-1]/12.
    #delta_2_x[-4,-8:]=np.array([2.,-27.,270.,-490.,270.,-27.,2.,0.])[::-1]/180.
    delta_2_x /= grid_size**2
    delta_2_x = csr(delta_2_x)
    delta_2_x.prune()
    
    global identity, csr_identity
    identity = np.eye(x_n+1, dtype=np.float64)
    csr_identity = csr(identity)
    global Simpson_integration
    # found to be not useful
    Simpson_integration = np.zeros_like(x)
    for i in range(x_n+1): Simpson_integration[i] = 2. if i%3==0 else 3.
    Simpson_integration[0] = 1.; Simpson_integration[-1] = 1.
    Simpson_integration *= 3./8.
    
    ################################### energy space start
    global quartic_V
    # \hbar = 1
    mass = 1./pi
    grid_size = x_max*2/x_n
    quartic_V = lambda_*x*x*x*x
    global p_hat, p_hat_2, x_2, xp_px_hat
    p_hat = -1.j*delta_x
    p_hat_2 = -1.*delta_2_x
    x_2 = x*x
    xp_px_hat = p_hat.dot(x_csr); xp_px_hat+=np.conj(xp_px_hat).T; xp_px_hat.prune()
    
    global quartic_Hamil, free_Hamil, wall_V, wall_Hmail, Hamil
    quartic_Hamil = csr(np.diag(quartic_V))+p_hat_2/(2.*mass)
    quartic_Hamil.prune()
    
    free_Hamil = p_hat_2/(2.*mass)
    
    wall_V = np.zeros_like(x); wall_V[-30:]=300
    wall_Hamil = csr(np.diag(wall_V))+p_hat_2/(2.*mass); wall_Hamil.prune()
    
    global upper4_diag, upper3_diag, upper2_diag, upper_diag, central_diag, lower_diag, lower2_diag, lower3_diag, lower4_diag
    Hamil = quartic_Hamil
    #Hamil = free_Hamil
    #Hamil = wall_Hamil
    array_Hamil = Hamil.toarray()
    upper2_diag = np.hstack(([0.,0.], np.diag(array_Hamil, k=2) )) *0.5j # -iH becomes iH after moving to the left
    lower2_diag = np.hstack((np.diag(array_Hamil, k=-2), [0.,0.])) *0.5j
    upper_diag = np.hstack(([0.], np.diag(array_Hamil, k=1) )) *0.5j
    lower_diag = np.hstack((np.diag(array_Hamil, k=-1),[0.])) *0.5j
    central_diag = np.diag(array_Hamil, k=0) *0.5j

    upper3_diag = np.hstack(([0.,0.,0.], np.diag(array_Hamil, k=3) )) *0.5j # -iH becomes iH after moving to the left
    lower3_diag = np.hstack((np.diag(array_Hamil, k=-3), [0.,0.,0.])) *0.5j
    upper4_diag = np.hstack(([0.,0.,0.,0.], np.diag(array_Hamil, k=4) )) *0.5j
    lower4_diag = np.hstack((np.diag(array_Hamil, k=-4), [0.,0.,0.,0.])) *0.5j
    return globals()
