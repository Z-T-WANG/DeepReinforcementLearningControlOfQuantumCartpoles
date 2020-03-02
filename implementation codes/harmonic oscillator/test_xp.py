#!/usr/bin/env python3
import numpy as np
from math import *
if __name__ != '__main__':
    from scipy.sparse import csr_matrix as csr
    from scipy.sparse import csc_matrix as csc
    from scipy import linalg
from time import time
import random
import torch
if __name__ == '__main__':
    import torch.multiprocessing as mp
#import matplotlib.animation as animation

################################### x space start (for display)

x_max = 14.; x_n = 250
x = np.linspace(-x_max,x_max,x_n, dtype=np.float64)

################################### x space end

################################### energy space start

omega = pi # don't ingore this energy multiplier !!!

def probability(state):
    return np.real(np.conj(state)*state)

def common_factor_of_1Dharmonics(n):
    return 1./np.sqrt(np.float128(repr(2**n))*np.float128(repr(factorial(n))))  *  sqrt(sqrt(1./pi)) * np.exp((-x*x/2).astype(np.float128))

def adjust_n_max(new_n_max):
    # \hbar = m\omega = 1; k = m\omega^2 = \omega
    global n_max
    n_max = new_n_max
    global n_phonon
    n_phonon=np.array([i for i in range(n_max+1)],dtype=np.float64)
    global sqrt_n
    sqrt_n = np.array([sqrt(i) for i in range(1,n_max+1)])
    global annihilation, creation
    annihilation=csr(np.diag(sqrt_n, k=1))
    creation=csr(np.diag(sqrt_n, k=-1))
    annihilation.prune(); creation.prune()
    global x_hat, p_hat
    x_hat = sqrt(1/2)*(creation + annihilation)
    p_hat = 1.j*sqrt(1/2)*(creation - annihilation)
    x_hat.prune(); p_hat.prune()
    global x_hat_2, p_hat_2, xp_px_hat
    x_hat_2 = x_hat.dot(x_hat); p_hat_2 = np.real(p_hat.dot(p_hat))
    xp_px_hat = x_hat.dot(p_hat)+p_hat.dot(x_hat)
    x_hat_2.prune(); p_hat_2.prune(); xp_px_hat.prune()
    global harmonic_Hamil
    harmonic_Hamil = omega *np.diag(1/2 + n_phonon)
    harmonic_Hamil = csr(harmonic_Hamil)
    harmonic_Hamil.prune()
    global eigen_states
    eigen_states=[]
    for i in range(new_n_max + 1):
        eigen_states.append(common_factor_of_1Dharmonics(i)*np.polynomial.hermite.hermval(x.astype(np.float128, order='C'), np.array([0. for j in range(i)]+[1.],dtype=np.float128)))
    eigen_states=np.array(eigen_states).transpose().astype(np.float64, order='C')
    print('n_max adjusted to {}'.format(new_n_max))

if __name__ != '__main__':
    adjust_n_max(130)

def normalize(vector):
    p=np.sum(probability(vector))
    return vector / sqrt(p)

def phonon_number(state):
    return np.sum(probability(state)*n_phonon)

def x_expct(state):
    return np.real(np.conj(state).dot(x_hat.dot(state)))

def p_expct(state):
    return np.real(np.conj(state).dot(p_hat.dot(state)))

def expct(state, hermitian_operator):
    return np.real(np.conj(state).dot(hermitian_operator.dot(state)))

def spatial_repr(state):
    mask = np.abs(state) > 1e-4
    return eigen_states[:, :state.size][:, mask].dot(state[ mask ])

def get_data(state):
    x_expc, p_expc = x_expct(state), p_expct(state)
    return x_expc, p_expc, expct(state, x_hat_2)-x_expc**2, expct(state, p_hat_2)-p_expc**2, expct(state, xp_px_hat)/2-x_expc*p_expc
################################### energy space end

################################## start learning setting

half_period_steps = 360*4 # 360
time_step = 1 / half_period_steps # ~= 0.003

controls_per_half_period = 18
control_interval = round(half_period_steps / controls_per_half_period) # 

num_of_episodes = 20000
reward_multiply = 0.1
failing_reward = -(80-20)*reward_multiply

n_periods_to_read = 5
read_length = n_periods_to_read * 2*half_period_steps # 3600
n_actions_to_read = n_periods_to_read * 2*controls_per_half_period

if __name__ == '__main__':
    import plot
    plot.set_parameters(x=x, x_max=x_max, dt=time_step, num_of_episodes=num_of_episodes, probability=probability, 
        reward_multiply=reward_multiply, read_length=read_length, controls_per_half_period=controls_per_half_period)

import RL
RL.set_parameters(control_interval=control_interval, reward_multiply=reward_multiply, failing_reward=failing_reward)

################################## end learning setting

class Control(object):
    """
    this class is used as a central controller that distributes information to workers, i.e. actions,
    and organizes when to call train, and record training loss.
    
    The structure of this program is:
    
    --> Control <--> {Do_Episode}_n   (multiprocessing actor, 
    |                            |    including respective plots, and also Step instances for caching
    |                            |
    | call & update actor        | store
    |                            |
    --> RL.Train <-- RL.Memory <--
    it contains the data that are related to AI's decisions and parameters for the epsilon strategy (EPS_ s),
    and a mutable gamma that controls the continuous measurement strength, which is used in Step.__call__
    because it remembers the statistics of the control of one episode, it needs to clear its statistics when 
    reinitialized, otherwise it would continue using its last action at the start of an episode
    """
    gamma = 1*omega
    EPS_START = 0.4
    EPS_END = 0.001
    EPS_DECAY = 4000
    def __init__(self, net):
        self.steps_done = 0
        self.net = net
        self.no_action_choice = net.num_of_control_resolution_oneside
        self.clear()
    def clear(self):
        self.accu_Loss = 0.
        self.cache = 0.
        self.last_action = self.no_action_choice
    def __call__(self, data):
        #eps_threshold = self.EPS_START * exp(-1. * self.steps_done / self.EPS_DECAY)
        #eps_threshold = max(0.05*self.EPS_START * exp(-1. * self.steps_done / (1000*self.EPS_DECAY)), eps_threshold)
        #eps_threshold = max(0.005*self.EPS_START * exp(-1. * self.steps_done / (4000*self.EPS_DECAY)), eps_threshold)
        #eps_threshold = max(self.EPS_END, eps_threshold)
        #self.steps_done += 1
        self.value = None
        data = torch.from_numpy(data).float().unsqueeze(0).cuda()
        with torch.no_grad():
            action_values, avg_value = self.net(data)
            action_values = action_values - action_values.mean(dim=1,keepdim=True) + avg_value.unsqueeze(1)
            value, last_action =action_values.max(1)
            value = value.item()
            self.last_action = last_action.item()
        force = self.net.convert_to_force(self.last_action)
        self.cache = force
        rnd = False
        return rnd
    def get_loss(self):
        return None
    def get_force(self):
        return self.cache

class Do_Episode(object):
    t_max = 100.
    def __init__(self, controller, state=None):
        self.step = Step()
        self.control = controller
        self.i_episode=0
        self.to_plot = False # switch of whether to plot or not
        if state.__class__ != type(None):
            self.state = state
            self.clear()
        else: self.reinit_state()
    def clear(self):
        self.measurement_results = np.array([],dtype=np.float32)
        print('clear state: Episode {}'.format(self.i_episode))
        self.cache_q = [0. for i in range(read_length)]
        self.t = 0.
        self.x_mean = [0. for i in range(read_length)]
        self.forces = [0. for i in range(read_length)]
        self.last_actions = [0. for i in range(n_actions_to_read)]
        self.last_action = self.control.no_action_choice
        self.to_stop = False
    def reinit_state(self):
        self.state = np.zeros((n_max+1,), dtype=np.complex128)
        self.state[0]=1.
        self.i_episode += 1
        self.clear()
        self.last_data = np.array(get_data(self.state))
    def __call__(self, n):
        if n_max+1 <= len(self.state): self.state = self.state[:n_max+1]
        elif n_max+1 > len(self.state): 
            self.state_new = np.zeros((n_max+1,))
            self.state_new[:len(self.state)] = self.state
            self.state = self.state_new
        if hasattr(self.step, 'ab'):
            self.step.ab = self.step.get_tridiagonal_matrix()
        if self.i_episode <= num_of_episodes:
            j=0
            self.to_stop = False
            stop = False
            force = self.control.get_force()
            gamma = self.control.gamma
            accu_energy = []; accu_counter = 0
            while j < n:
                m = self.measurement_results
                if m.size + len(self.cache_q) >= read_length + control_interval:
                    self.data = np.array(get_data(self.state))
                    m = np.hstack((m, np.array(self.cache_q, dtype=np.float32)))
                    self.last_actions.append(-1.*force)
                    if not self.to_stop: 
                         pass
                    else: 
                        stop = True
                    m = m[-read_length:] # reload the measurement data results
                    self.measurement_results = m
                    self.last_actions = self.last_actions[-n_actions_to_read:]
                    self.cache_q=[]
                    rnd = self.control(self.data)
                    force = self.control.get_force()
                    self.last_action = self.control.last_action
                    value = self.control.value
                    self.last_data = self.data
                    if self.t>40: 
                        if self.t%10<time_step: accu_energy.append(phonon_number(self.state)); accu_counter += 1
                self.state, q, x_mean, Fail = self.step(self.state, time_step, force, gamma)
                self.cache_q.append(q)
                if Fail and not self.to_stop : self.to_stop = True 
                self.t += time_step
                if stop or self.t >= self.t_max:
                    if not stop: print('\nSucceeded')
                    else: 
                        stop = False; print('\nFailed \t t = {:.2f}'.format(self.t))
                    self.reinit_state()
                    j += 1
        energy_array = np.array(accu_energy)
        mean, std = np.mean(energy_array), sqrt(np.var(energy_array, ddof=1))
        print( '{:.7f} +- {:.7f}'.format(mean, std) )
        return mean, std
    def do_plot(self):
        self.clear_recorded_Xmean_forces()
        if self.to_plot: self.plot(spatial_repr(self.state), np.hstack(( self.measurement_results, np.array( self.cache_q ) ))[-read_length:], self.x_mean, self.forces)

if __name__ != '__main__':
    class Step(object):
        def get_tridiagonal_matrix(self):
            temp_array = self.dt_cache*(0.5j)*omega*self.force_cache*sqrt(1/2)*sqrt_n
            upper_diag = np.hstack(([0.],temp_array))
            lower_diag = np.hstack((temp_array,[0.]))
            diag = np.full_like(upper_diag, 1.) + self.dt_cache*(0.5j)*np.diag(harmonic_Hamil.toarray())
            Hamiltonian = (harmonic_Hamil+omega*self.force_cache*x_hat)
            self.Hamiltonian_square = Hamiltonian.dot(Hamiltonian)
            return np.vstack((upper_diag,diag,lower_diag))
        def D1(self, state, x_avg=None):
            if x_avg==None: x_avg=x_expct(state)
            x_hat_state=x_hat.dot(state)
            relative_state=x_hat_state-x_avg*state
            return (-1.j)*(harmonic_Hamil.dot(state) + omega*self.force_cache*x_hat_state) - self.gamma/4*(x_hat.dot(relative_state)-x_avg*relative_state), relative_state
        def D1ImRe(self, state, x_avg=None):
            if x_avg==None: x_avg=x_expct(state)
            x_hat_state=x_hat.dot(state)
            relative_state=x_hat_state-x_avg*state
            return (-1.j)*(harmonic_Hamil.dot(state) + omega*self.force_cache*x_hat_state), - self.gamma/4*(x_hat.dot(relative_state)-x_avg*relative_state), relative_state
        def D2(self, state, relative_state=None):
            if relative_state.__class__==type(None): relative_state = x_hat.dot(state) - x_expct(state)*state
            return sqrt(self.gamma/2)*relative_state
        def __call__(self, state, dt, force, gamma): 
            # [Kloeden and Platen, Numerical Solution of Stochastic Differential Equations, p.408, (3.4), implicit strong order 1.5 for the imaginary part, and p.378, (2.1)]
            if not hasattr(self, 'dt_cache'):
                self.dt_cache=dt; self.force_cache=force
                self.ab = self.get_tridiagonal_matrix()
            elif dt != self.dt_cache or force != self.force_cache:
                self.dt_cache=dt; self.force_cache=force
                self.ab = self.get_tridiagonal_matrix()
            self.gamma = gamma

            # Eq. (10.4.3): initialize random variables
            U1,U2=np.random.normal(size=(2,))
            dW = sqrt(dt)*U1; dZ = (sqrt(dt)*dt)*0.5*(U1+U2/sqrt(3))

            x_mean = x_expct(state)
            q = x_mean + dW/sqrt(2*gamma)/dt
            D1_state, relative_state = self.D1(state, x_mean)
            D2_state = self.D2(state, relative_state)
            D2_state_dW, D2_state_drt = D2_state*dW, D2_state*sqrt(dt)
            Y = state + D1_state*dt
            Y_plus, Y_minus = Y + D2_state_drt, Y - D2_state_drt # these terms have normalization error up to 1st order of dt
            D1_Y_plusIm, D1_Y_plusRe, relative_Y_plus = self.D1ImRe(Y_plus)
            D1_Y_minusIm, D1_Y_minusRe, relative_Y_minus = self.D1ImRe(Y_minus)
            D2_Y_plus, D2_Y_minus = self.D2(Y_plus,relative_Y_plus), self.D2(Y_minus,relative_Y_minus)
            Phi_plus, Phi_minus = Y_plus + sqrt(dt)*D2_Y_plus, Y_plus - sqrt(dt)*D2_Y_plus
            D2_Phi_plus, D2_Phi_minus = self.D2(Phi_plus), self.D2(Phi_minus)
            D1_Y_plusIm_substract_D1_Y_minusIm = D1_Y_plusIm-D1_Y_minusIm

            # Start Eq. (11.2.1)
            # The 6th line is the 4th line of Eq. (12.3.4). This term cancels the b*dW term in the implicit term of state', as an 3/2 order correction, since state' appears as state' * dt
            state = state + D2_state_dW + 0.5/sqrt(dt)*dZ*(D1_Y_plusIm_substract_D1_Y_minusIm + D1_Y_plusRe-D1_Y_minusRe) + \
                0.25*dt*(D1_Y_plusRe+2*D1_state+D1_Y_minusRe) + \
                0.25/sqrt(dt)*(dW*dW-dt)*(D2_Y_plus - D2_Y_minus) + \
                0.5/dt*(dW*dt-dZ)*(D2_Y_plus + D2_Y_minus - 2*D2_state) + \
                0.25/dt*(dW*dW/3-dt)*dW*(D2_Phi_plus - D2_Phi_minus - D2_Y_plus + D2_Y_minus) \
              - 0.25*sqrt(dt)*dW*(D1_Y_plusIm_substract_D1_Y_minusIm) \
              + dt*dt*dt/12.*self.Hamiltonian_square.dot(D1_state)
            # a third order correction term of Hamiltonian*dt, since its error results in decrease of high energy components
            # the coefficient -1/12 comes from the fact that implicit term D1Im(state') includes a1'*(a'a*(dt^2/2)), which becomes
            # (dt^3/4)(a1'*a'*a), larger than (dt^3/6)(a1'*a1'*a) by a factor of (dt^3/12)
            if np.absolute(state[-11]) < 1e-5: Fail = False
            else: Fail = True
            state = linalg.solve_banded((1,1), self.ab, state, overwrite_ab=False, overwrite_b=True)
            state = normalize(state)
            return state, q, x_mean, Fail

def test(args):
    net, n, seed = args
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    call_control=Control(net.cuda())
    do_episode = Do_Episode(call_control)
    do_episode.to_plot = False
    return do_episode(n)

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    nets = []
    for i in range(1,17):
        nets.append( torch.load( './DQN_save/{}'.format(i) ) )
    with mp.Pool(processes=8) as pool:
        results = list(pool.imap(test, [(net,1000,random.randrange(0,9999999)) for net in nets]))
    for i,result in enumerate(results):
        print('number {}: {}/{}'.format(i, result[0],result[1]))
