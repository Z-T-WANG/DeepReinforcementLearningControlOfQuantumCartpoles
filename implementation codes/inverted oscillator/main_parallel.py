#!/usr/bin/env python3.6
import numpy as np
from math import pi, sqrt, exp
from scipy.sparse import csr_matrix as csr
from scipy import linalg
import random
import torch
import os, sys
import time
from termcolor import colored
import copy
if __name__ == '__main__':
    from math import *
    import torch.multiprocessing as mp
    import fnmatch
    import argparse
    from multiprocessing.sharedctypes import Value, RawValue
    #import matplotlib.animation as animation


# the commandline arguments are detailed in "arguments.py" because there are too many
from arguments import args
torch.cuda.set_device(args.gpu_id)
torch.set_num_threads(1) # this is for CPU usage
if args.seed != -1: 
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

args.omega = 1.


################################### x space (for visulization)

x_max = 14.; x_n = 250
x = np.linspace(-x_max,x_max,x_n, dtype=np.float64)

################################### x space end

################################### wave function and energy


omega = args.omega * pi # this is the energy multiplier for the phonon numbers,
                        # in units of time^{-1} or \omega_c
gamma = args.gamma * pi
n_max = args.n_max

def probability(state):
    return np.real(np.conj(state)*state)

def common_factor_of_1Dharmonics(n):
    return 1./np.sqrt(np.float128(repr(2**n))*np.float128(repr(factorial(n))))  *  sqrt(sqrt(1./pi)) * np.exp((-x*x/2).astype(np.float128))

def adjust_n_max(new_n_max):
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
    global x_hat, p_hat  ### we assume \hbar = m * \omega = 1
    x_hat = sqrt(1/2)*(creation + annihilation)
    p_hat = 1.j*sqrt(1/2)*(creation - annihilation)
    x_hat.prune(); p_hat.prune()
    global x_hat_2, p_hat_2, xp_px_hat
    x_hat_2 = x_hat.dot(x_hat); p_hat_2 = np.real(p_hat.dot(p_hat))
    xp_px_hat = x_hat.dot(p_hat)+p_hat.dot(x_hat)
    x_hat_2.prune(); p_hat_2.prune(); xp_px_hat.prune()
    global harmonic_Hamil
    harmonic_Hamil = (-1/2 * omega * (creation.dot(creation) + annihilation.dot(annihilation))).toarray()
    global upper2_diag
    global lower2_diag
    upper2_diag = np.hstack(([0.,0.], np.diag(harmonic_Hamil, k=2) )) *1.j*0.5
    lower2_diag = np.hstack((np.diag(harmonic_Hamil, k=-2), [0.,0.])) *1.j*0.5
    harmonic_Hamil = csr(harmonic_Hamil)
    harmonic_Hamil.prune()
    if __name__ == '__main__':
        print('n_max adjusted to {}'.format(new_n_max))
        global eigen_states
        eigen_states=[]
        for i in range(new_n_max + 1):
            eigen_states.append(common_factor_of_1Dharmonics(i)*np.polynomial.hermite.hermval(x.astype(np.float128, order='C'), np.array([0. for j in range(i)]+[1.],dtype=np.float128)))
        eigen_states=np.array(eigen_states).transpose().astype(np.float64, order='C')

def normalize(vector):
    p=linalg.norm(vector)
    return vector / p

def x_expct(state):
    return np.real(np.conj(state).dot(x_hat.dot(state)))

def p_expct(state):
    return np.real(np.conj(state).dot(p_hat.dot(state)))

def expct(state, hermitian_operator):
    return np.real(np.conj(state).dot(hermitian_operator.dot(state)))

def spatial_repr(state):
    mask = np.abs(state) > 1e-4 # only values that are larger than this threshold are displayed
    return eigen_states[:, :state.size][:, mask].dot(state[ mask ])

if __name__ != '__main__':
    adjust_n_max(n_max)

################################### wave function and energy end


################################## start learning setting, including some other settings not present in the commandline arguments

half_period_steps = args.time_steps
time_step = 1 / half_period_steps

controls_per_half_period = args.n_con
assert half_period_steps % controls_per_half_period == 0
control_interval = round(half_period_steps / controls_per_half_period)

num_of_episodes = args.num_of_episodes
alive_reward = args.alive_reward
failing_reward = -1.


num_of_saves = args.num_of_saves


# data input for the neural network
def get_data_xp(state):
    x_expc, p_expc = x_expct(state), p_expct(state)
    return np.array([x_expc, p_expc, expct(state, x_hat_2)-x_expc**2, expct(state, p_hat_2)-p_expc**2, expct(state, xp_px_hat)/2-x_expc*p_expc]).astype(np.float32)

def get_data_wavefunction(state):
    # the last five values at the highest levels are supposed negligible and we do not include them as input data for AI
    return np.hstack((np.real(state[:-20]),np.imag(state[:-20]))).astype(np.float32) 

if args.input == 'xp':
    data_size = 5
    get_data = get_data_xp    # get_data is the function that returns the input data for AI given a quantum state
elif args.input == 'wavefunction':
    data_size = 2*(n_max+1-20)
    get_data = get_data_wavefunction
elif args.input == 'measurements':
    n_periods_to_read = 2.
    num_of_data_per_time_unit = 360*4
    assert half_period_steps % num_of_data_per_time_unit == 0
    coarse_grain = half_period_steps//num_of_data_per_time_unit
    read_length = round(n_periods_to_read * 2 * num_of_data_per_time_unit) # 3600
    read_control_step_length = control_interval//coarse_grain
    data_size = 2 * read_length
    shape_measurement_data = (2, read_length)



# we do not plot when we do parallelized computation

#import plot
#plot.set_parameters(x=x, x_max=x_max, dt=time_step, num_of_episodes=num_of_episodes, probability=probability, 
#    reward_multiply=reward_multiply, read_length=read_length, controls_per_half_period=controls_per_half_period)


# set the reinforcement learning settings
if __name__ == '__main__':
    import RL
    RL.set_parameters(control_interval=control_interval, failing_reward=failing_reward, F_max=args.F_max)
    if args.input == 'measurements': RL.set_parameters(read_step_length=read_control_step_length)

################################## end learning setting


# Below is the worker function for subprocesses, which carries out the control simulations and pushes the experiences and records to queues that are collected and handled by other processes. (Quantum simulation is implemented in a compiled C module)
# Because too many processes using CUDA will occupy a huge amount of GPU memory, we avoid using CUDA in these workers. Instead, these workers ask a manager process when they want to evaluate the neural network, and only the manager process is allowed to use CUDA to evaluate the neural network for the controls.
def Control(net, pipes, shared_buffer, seed, idx):
    simulation = __import__('simulation')
    # seeding
    random.seed(seed)
    np.random.seed(seed)
    simulation.set_seed(seed)
    # preparing pipes
    MemoryQueue, ResultsQueue, ActionPipe, EndEvent, PauseEvent = pipes
    state_data_to_manager = np.frombuffer(shared_buffer,dtype='float32')
    if args.input=='measurements': state_data_to_manager = state_data_to_manager.reshape(shape_measurement_data)
    # random action decision hyperparameters
    EPS_START = 0.1
    EPS_END = 0.002
    EPS_DECAY = args.n_con*100*300
    # initialization
    steps_done = 0
    no_action_choice = net.num_of_control_resolution_oneside
    xth = net.convert_to_force(2*no_action_choice)
    def call_force(data):
        nonlocal steps_done
        # if LQG control is used, immediately return the LQG control without evaluating the neural network.
        if args.LQG:
            x=data[0]; p=data[1]
            rnd = False
            # the "dt" here is the time of one control force step
            dt = 1./controls_per_half_period
            force_max = net.convert_to_force(2*no_action_choice)
            F = - (x+p)*(1+omega*dt+0.5*omega*omega*dt*dt)/(dt+omega*dt*dt/2)
            force = F / omega
            force = min(force, force_max)
            force = max(force, -force_max)
            force = round(force/(force_max/no_action_choice))
            force = float(force*(force_max/no_action_choice))
            return force, round(force/(force_max/no_action_choice))+no_action_choice, False

        # apply an \epsilon-greedy strategy:
        eps_threshold = (EPS_START-EPS_END) * exp(-1. * steps_done / EPS_DECAY)
        eps_threshold += EPS_END
        steps_done += args.num_of_actors # this approximates the total steps_done of all the actors
        if random.random() < eps_threshold and not args.test:
            last_action=random.randrange(no_action_choice*2+1) 
            rnd = True
        else:
            # copy data to 
            state_data_to_manager[:]=data
            while ActionPipe.poll(): ActionPipe.recv() # ensure that no data remain in the recv pipe
            ActionPipe.send(idx)
            last_action = ActionPipe.recv()
            rnd = False
        force = net.convert_to_force(last_action)
        return force, last_action, rnd

    state = np.empty((n_max+1,), dtype=np.complex128)
    # do one episode
    def do_episode():
        t = 0.
        to_stop = False
        # prepare the quantum state
        state[:] = 0.
        state[0] = 1.
        # force is the parameter before -\omega\hat{x}, which is the physical force divided by an omega factor
        force = 0.
        last_action = no_action_choice
        # start the simulation loop
        i = 0
        experience = []
        numerical_failure = False
        if args.input!='measurements':
            last_data = get_data(state)*args.input_scaling
            while True:
                if i % control_interval == 0 and i != 0:
                    if to_stop:
                        numerical_failure = True
                    else: to_stop = (abs(simulation.x_expectation(state)) > xth)
                    if not to_stop:
                        data = get_data(state)*args.input_scaling
                        if args.train:
                            experience.append(np.hstack(( last_data, data, 
                                np.array([last_action],dtype=np.float32),  
                                np.array([1.],dtype=np.float32) )) ) # the alive_reward is stored and used directly in class Train() 
                    else:
                        if args.train:
                            experience.append(np.hstack(( last_data, data, 
                                np.array([last_action],dtype=np.float32), 
                                np.array([failing_reward],dtype=np.float32) )) )
                        break
                    if (not args.LQG) or args.input=='xp':
                        force, last_action, rnd = call_force(data) 
                    # use "get_data_xp" only when args.LQG==True and args.input!='xp'
                    else: force, last_action, rnd = call_force(get_data_xp(state))
                    last_data = data
                q, x_mean, Fail = simulation.step(state, time_step, force, gamma)
                i += 1
                # to_stop tiggers the stop when it attempts to store experience
                if Fail and not to_stop : to_stop = True
                t += time_step
        # the organization of measurement data needs to be different:
        else:
            measurements_cache = []; measurements_input = list(np.zeros(read_length))
            forces_along_measurements_input = list(np.zeros(read_length)); forces_to_store = list(np.zeros(read_length//read_control_step_length))
            while True:
                if i % control_interval == 0 and i != 0:
                    if to_stop:
                        numerical_failure = True
                    else: to_stop = (abs(simulation.x_expectation(state)) > xth)
                    forces_to_store.append(force*args.input_scaling)
                    if not to_stop:
                        # store the experience as a continuous measurement sequence connecting two neighbouring control steps
                        if args.train and i!=control_interval: 
                            experience.append(np.hstack(( np.array(measurements_input, dtype=np.float32)[::-1],
                                np.array(forces_to_store, dtype=np.float32)[::-1],
                                np.array([last_action], dtype=np.float32), 
                                np.array([1.],dtype=np.float32) )) ) # the alive_reward is stored and used directly in class Train() 
                    else:
                        if args.train and i!=control_interval:
                            experience.append(np.hstack(( np.array(measurements_input, dtype=np.float32)[::-1],
                                np.array(forces_to_store, dtype=np.float32)[::-1], 
                                np.array([last_action],dtype=np.float32), 
                                np.array([failing_reward],dtype=np.float32) )) )
                        break
                    # organise the lists to discard measurement data that belong to the most distant control step in the past
                    measurements_input, forces_along_measurements_input = measurements_input[read_control_step_length:], forces_along_measurements_input[read_control_step_length:]
                    forces_to_store = forces_to_store[1:]
                    # use the organised measurement data to compute the next control
                    if (not args.LQG):
                        force, last_action, rnd = call_force(np.array([measurements_input[::-1], forces_along_measurements_input[::-1]]))
                    else: force, last_action, rnd = call_force(get_data_xp(state))
                q, x_mean, Fail=simulation.step(state, time_step, force, gamma)
                measurements_cache.append(q) 
                if len(measurements_cache)==coarse_grain: 
                    measurements_input.append(sum(measurements_cache)/coarse_grain*args.input_scaling)
                    measurements_cache.clear()
                    forces_along_measurements_input.append(force*args.input_scaling) # we rescale the force by its duration  
                i += 1
                # to_stop tiggers stop when it stores experience
                if Fail and not to_stop : to_stop = True
                t += time_step
        # push experience into the main process and push results to the manager
        if not EndEvent.is_set():
            MemoryQueue.put( (experience, t, numerical_failure) )
            ResultsQueue.put((t,))
        return t
    while True:
        # whether to end the program
        if EndEvent.is_set():
            break
        do_episode()
        while PauseEvent.is_set():
            time.sleep(1.)
            # to avoid an endless loop
            if EndEvent.is_set():
                break
    while ActionPipe.poll(): ActionPipe.recv() # ensure that no data remain in the recv pipe
    ActionPipe.send(None) # tell the manager that the worker has ended
    return


# the manager process for workers. It is used to organise all neural network evaluations into one single process in order to save GPU memory.
# It also monitors the current performance and saves the models.
def worker_manager(net, pipes, num_of_processes, seed, others):
    # initialize
    MemoryQueue, ActorPipe, EndEvent, PauseEvent = pipes
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    # prepare the path
    if not os.path.isdir(args.folder_name): os.makedirs(args.folder_name, exist_ok=True)
    # prepare workers
    import multiprocessing as mp
    from multiprocessing.sharedctypes import RawArray
    fork = mp.get_context('forkserver')
    results_queue = fork.Manager().Queue()
    processes = []
    message_conn = []; message_worker_conn = []
    worker_data = []
    for n in range(num_of_processes):
        conn1, conn2 = fork.Pipe(True)
        message_conn.append(conn1); message_worker_conn.append(conn2)
        shared_buffer = RawArray('f', data_size)
        np_memory = np.frombuffer(shared_buffer,dtype='float32')
        if args.input=='measurements': 
            np_memory=np_memory.reshape(shape_measurement_data)
        worker_data.append(torch.from_numpy(np_memory))
        seed = random.randrange(0,99999)
        processes.append( fork.Process( target=Control, args=(copy.deepcopy(net).cpu(), (MemoryQueue, results_queue, conn2, EndEvent, PauseEvent), shared_buffer, seed, n) ) )
    net=net.cuda()
    net.eval()
    # prepare to save
    save_models=[]
    simulated_oscillations = 0.
    episode_passed = 0
    # for cartpole tasks, the variance of estimated performance is too high. So we give up evaluating the performances during training.
    if args.test or args.LQG: performances = []
    # when receiving a net, check whether the previous net should be stored
    def receive_net():
        if args.test or args.LQG:
            if not results_queue.empty():
                while not results_queue.empty():
                    result = results_queue.get()
                    performances.append(result[0])
                    with open(os.path.join(args.folder_name, others+'_record.txt'),'a') as f:
                        f.write('{}\n'.format(result[0]))
            return

        if ActorPipe.poll():
            nonlocal net
            if not results_queue.empty():
                nonlocal episode_passed, simulated_oscillations
                while not results_queue.empty():
                    result = results_queue.get()
                    simulated_oscillations += result[0]/2.
                    episode_passed += 1
                    if args.write_training_data:
                        with open(args.folder_name + '.txt','a') as f:
                            f.write('{}, {}\n'.format(simulated_oscillations, result[0]))
                # if 300 additional episodes have passed, we save the next model
                if episode_passed >= args.save_interval:
                    nonlocal save_models
                    save_models.append(copy.deepcopy(net.state_dict()))
                    save_models = save_models[-num_of_saves:]
                    episode_passed = 0
                    for idx, model in enumerate(reversed(save_models)):
                        torch.save(model, os.path.join(args.folder_name,'{}.pth'.format(idx+1)))
                        existing_record_name = os.path.join(args.folder_name,'{}_record.txt'.format(idx+1))
                        if os.path.isfile(existing_record_name): os.remove(existing_record_name)
            net.load_state_dict(ActorPipe.recv())
            while ActorPipe.poll():
                net.load_state_dict(ActorPipe.recv())
            net = net.cuda()
            net.eval()
            if args.show_actor_recv: print(colored('new model received', 'yellow'))

    if args.input != 'measurements':
        network_input = torch.empty((num_of_processes,data_size), device='cuda')
    else: network_input = torch.empty((num_of_processes,2,read_length), device='cuda')

    for proc in processes:
        proc.start()
    process_ended = 0
    while process_ended!=num_of_processes:
        # receive network input data (confirm that there are data, and copy them from the shared buffer to GPU)
        num_of_data = 0
        data_received_id = []
        for i in range(num_of_processes):
            if message_conn[i].poll():
                idx=message_conn[i].recv()
                if idx!=None:
                    data_received_id.append(idx) # the data sent through pipe is exactly the id
                    network_input[num_of_data]=worker_data[i]
                    num_of_data += 1
                else: process_ended += 1

                while message_conn[i].poll(): # For safety; when False, this while loop is ignored
                    message_conn[i].recv()

        # process the received data
        if num_of_data == 0:
            time.sleep(0.0005) # if no data, wait for 0.5 milisecond
            if args.LQG: time.sleep(1.)
        else:
            action_values, avg_value, _noise = net(network_input[:num_of_data,:])
            actions = action_values.max(1)[1].cpu()
            for i,idx in enumerate(data_received_id):
                message_conn[idx].send(actions[i].item())
        # update the network
        receive_net()
    # end, if have left the while loop
    if args.test or args.LQG:
        with open(os.path.join(args.folder_name, others+'.txt'),'w') as f:
            performances = np.array(performances)
            f.write('{} +- {}\n'.format(np.mean(performances), np.std(performances, ddof=1)/np.sqrt(len(performances)) ))
    for proc in processes:
        proc.join()

if __name__ == '__main__':
    class Main_System(object):
        # the Main_System does not need to keep a copy of network
        # only the copy of network inside TrainDQN class is modified by training, so we pass its state_dict to subprocesses
        def __init__(self, train, num_of_processes, others=''):
            self.train = train
            self.processes = []
            self.actor_update_time = 10.
            self.lr_step = 0
            self.pending_training_updates = Value('d',0.,lock=True)
            # somehow RawValue also needs us to call ".value", otherwise it says the type is c_double or c_int
            self.episode = RawValue('i',0)
            self.t_done = Value('d',0.,lock=True)
            self.last_achieved_time = RawValue('d',0.)
            # set the data going to subprocesses:
            self.train.memory.start_proxy_process((self.pending_training_updates, self.episode, self.t_done, self.last_achieved_time), self.train.transitions_storage, (self.train.batch_size, self.train.memory.tree.data_size))
            # the following will create threads, which not end and cause error (not exiting) 
            spawn=mp.get_context('spawn')
            self.manager = spawn.Manager()
            self.MemoryInputQueue = self.manager.Queue()
            self.end_event = self.manager.Event()
            self.pause_event = self.manager.Event()
            self.learning_in_progress_event = self.manager.Event()
            # actors
            self.ActorReceivePipe, self.ActorUpdatePipe = spawn.Pipe(False) # unidirectional pipe that send message from conn2 to conn1
            seed = random.randrange(0,9999999)
            self.worker_manager = spawn.Process( target=worker_manager, args=(copy.deepcopy(train.net).cpu(), (self.MemoryInputQueue, self.ActorReceivePipe, self.end_event, self.pause_event), num_of_processes, seed, others) )
            # store and manage experience (including updating priority and potentially sampling out replays)
            # all the arguments passed into it are used (**by fork initialization in RL module**).
            # somehow RawValue also needs us to call ".value" ? Otherwise it says the type is c_double / c_int
            self.train.memory.set_memory_source(self.MemoryInputQueue, (self.pause_event, self.end_event, self.learning_in_progress_event))
            self.backup_period = self.train.backup_period
            self.train.backup_period = 100
        def __call__(self, num_of_episodes):
            started = False
            self.worker_manager.start()
            last_time = time.time()
            last_idle_time = 0.
            updates_done = 0.
            # We assume batch_size is 256 and each experience is learned 8 times in RL.py, and when we change, we use the rescaling factor below to implement.
            # If we disable training, we use 'inf' instead to make the condition of training always False.  
            downscaling_of_default_num_updates = (8./args.n_times_per_sample)*(args.batch_size/256.) if args.train else float('inf')

            while self.episode.value < num_of_episodes or (self.episode.value < args.maximum_trails_before_giveup and not self.learning_in_progress_event.is_set()):
                something_done = False # check whether nothing is done in one event loop
                remaining_updates = self.pending_training_updates.value - updates_done
                if remaining_updates >= 1. *downscaling_of_default_num_updates:
                    if remaining_updates >= 150. *downscaling_of_default_num_updates and not self.pause_event.is_set():
                        self.pause_event.set(); print('Wait for training')
                    loss = self.train()
                    # if we parallelize the training as a separate process, the following block should be deleted
                    if loss!=None:
                        updates_done += 1.*downscaling_of_default_num_updates
                        something_done = True # one training step is done
                        if not started: started = True
                        # to reduce the frequency of calling "get_lock()", we only periodically reset the shared data "pending_training_updates"  
                        if updates_done >= 200.*downscaling_of_default_num_updates:
                            with self.pending_training_updates.get_lock():
                                self.pending_training_updates.value -= updates_done
                                updates_done = 0.
                if remaining_updates < 50. *downscaling_of_default_num_updates and self.pause_event.is_set():
                    self.pause_event.clear()
                if self.t_done.value >= self.actor_update_time:
                    self.scale_up_actor_update_time(self.last_achieved_time.value)
                    if not self.ActorReceivePipe.poll() and started and not args.LQG:
                        self.ActorUpdatePipe.send(self.train.net.state_dict())
                        with self.t_done.get_lock():
                            self.t_done.value = 0.
                        something_done = True
                if something_done:
                    # print out how much time the training process has been idle for
                    if last_idle_time != 0. and time.time() - last_time > 50.: 
                        print('trainer pending for {:.1f} seconds out of {:.1f}'.format(last_idle_time, time.time() - last_time))
                        last_idle_time = 0.
                        last_time = time.time()
                # if nothing is done, wait.
                if not something_done: time.sleep(0.01); last_idle_time += 0.01
                self.adjust_learning_rate()
            self.end_event.set()
            self.worker_manager.join()
            if not self.learning_in_progress_event.is_set(): 
                print('\nLearning is not in progress and we have tried {} times. Exit.'.format(args.maximum_trails_before_giveup))
            return
        def scale_up_actor_update_time(self, achieved_time):
            changed = False
            if achieved_time>50. and self.actor_update_time<=150.:
                self.actor_update_time = 800.; changed = True
            elif achieved_time>20. and self.actor_update_time<=50.:
                self.actor_update_time = 150.; changed = True
            elif achieved_time>10. and self.actor_update_time<=25.:
                self.actor_update_time = 50.; changed = True
                #for param_group in self.train.optim.param_groups: param_group['lr'] = 4e-4
            elif achieved_time>5. and self.actor_update_time<=10.:
                self.actor_update_time = 25.; changed = True
            if changed and args.train: print('actor_update_time adjusted to {:.1f}'.format(self.actor_update_time))
        def adjust_learning_rate(self):
            # the learning rate schedule is written in "arguments.py"
            if self.train.backup_period != self.backup_period and self.learning_in_progress_event.is_set():
                self.train.backup_period = self.backup_period
            if self.episode.value > args.lr_schedule[self.lr_step][0] and self.learning_in_progress_event.is_set():
                args.lr = min(args.lr_schedule[self.lr_step][1], args.lr)
                self.lr_step += 1
                if args.train:
                    for param_group in self.train.optim.param_groups: param_group['lr'] = args.lr
                    print(colored('learning rate set to {:.2g}'.format(args.lr),attrs=['bold']))
                    if self.lr_step == 1: self.train.gamma = 0.998 # change the reinforcement learning gamma when we first decay the learning rate



# system settings, checks and the framework
def check_C_module_and_compile():
    if args.compile == False:
        try:
            simulation = __import__('simulation')
            (compiled_n_max, compiled_omega) = simulation.check_settings()
            if compiled_n_max != n_max:
                print(colored('N_MAX of the existing C module ({}) does not match the current task ({}). Recompile.\n'.format(compiled_n_max, n_max), 'yellow',attrs=['bold']))
                time.sleep(1)
                args.compile = True
            elif compiled_omega != omega:
                print(colored('\omega of the existing C module ({}) does not match the current task ({}). Recompile.\n'.format(compiled_omega, omega), 'yellow',attrs=['bold']))
                time.sleep(1)
                args.compile = True
            else: print("\ncompiled simulation module is loaded") 
        except (ModuleNotFoundError, AttributeError) as e:
            args.compile = True
    if args.compile == True:
        code=os.system('python{} setupC.py --n_max {} --omega {} --gamma {}'.format(sys.version[:3], n_max, omega, gamma))
        if code != 0:
            raise RuntimeError('Compilation Failure')

if __name__ == '__main__':
    time_of_start = time.time()
    # set the title of the terminal so that what the terminal is doing is clear
    print('\33]0;{}\a'.format(' '.join(sys.argv)), end='', flush=True)
    print(args)

    # compile the simulation module in C
    check_C_module_and_compile()

    # set the replay memory
    capacity = round(args.size_of_replay_memory*controls_per_half_period*100.) if args.train else 1
    memory = RL.Memory(capacity = capacity, data_size = data_size * 2 + 2 if args.input != 'measurements' else \
                                            (read_control_step_length+read_length) + read_length//read_control_step_length+1 + 2,
                                            policy = 'random', passes_before_random = 0.2)
    # define the neural network
    net = RL.direct_DQN(data_size).cuda() if args.input != 'measurements' else RL.DQN_measurement(read_length)
    # set the task
    if args.train or args.LQG:
        train = RL.TrainDQN(net, memory, batch_size = args.batch_size, gamma=0.99, backup_period = args.target_network_update_interval, args=args)
        del net
        # the main function of training
        if args.train: 
            main = Main_System(train, num_of_processes=args.num_of_actors)
            main(num_of_episodes)
        # when we do not train and we test the result of LQG
        elif args.LQG: 
            main = Main_System(train, num_of_processes=args.num_of_actors, others='LQG')
            main(args.num_of_test_episodes)
    # if we test existing models, we use a loop to iterate over the models
    else:
        # find all models to test that end with no extension or '.pth' in the given directory
        import glob
        test_nets = []
        for name in glob.glob(os.path.join(args.folder_name,'*')):
            file_name, ext = os.path.splitext(os.path.basename(name))
            if ext=='.pth' or ext=='': test_nets.append((file_name, torch.load(name)))
        assert len(test_nets)!=0, 'No model found to test'
        # for each model we run the main loop once
        for test_net in test_nets:
            net.load_state_dict(test_net[1])
            train = RL.TrainDQN(net, memory, batch_size = args.batch_size, gamma=0.99, backup_period = args.target_network_update_interval, args=args)
            main = Main_System(train, num_of_processes=args.num_of_actors, others=test_net[0])
            main(args.num_of_test_episodes)
        del net
        # organize all test results into one file
        with open(os.path.join(args.folder_name,'test_result.txt'),'w') as test_result:
            for test_net in test_nets:
                with open(os.path.join(args.folder_name,test_net[0]+'.txt')) as f:
                    result = f.readline()
                    test_result.write('{}:\t'.format(test_net[0])+result)
                    print('{}:\t'.format(test_net[0])+result, end='')
                os.remove(os.path.join(args.folder_name,test_net[0]+'.txt'))
    del main
    del memory
    from timer import print_elapsed_time
    print_elapsed_time(time_of_start)
