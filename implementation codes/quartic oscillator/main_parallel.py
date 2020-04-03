#!/usr/bin/env python3
import os, sys
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from math import pi, sqrt, exp
from scipy.sparse import csr_matrix as csr
from scipy import linalg
import random
import torch
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

lambda_=args.__dict__['lambda']*pi
gamma = args.gamma*pi
mass = args.mass/pi
x_max, x_n = args.x_max, args.x_n

# the global settgins are written in the file "space_def.py" as the output (dictionary) of the function "set_global"
# this global definition strategy is not appreciated, but otherwise it would be annoyingly long
import space_def
globals().update({k:v for k,v in space_def.set_global(x_max = x_max, x_n_ = x_n, lambda_ = lambda_, mass = mass).items() if k not in globals()})

def probability(state):
    return np.real(np.conj(state)*state)

def x_expct(state):
    return np.real(np.conj(state).dot(x*state))*grid_size

def x_2_expct(state):
    return np.real(np.conj(state).dot(x_2*state))*grid_size

def kurtosis(state):
    x_mean=x_expct(state)
    relative_x = x-x_mean
    relative_x_2=relative_x*relative_x
    return np.real(np.conj(state).dot(relative_x_2*relative_x_2*state))*grid_size/(np.real(np.conj(state).dot(relative_x_2*state))*grid_size)**2 - 3

def p_expct(state):
    return np.real(np.conj(state).dot(p_hat.dot(state)))*grid_size

def xpx_expct(state):
    return np.real(np.conj(state).dot(x*p_hat.dot(x*state)))*grid_size

def expct(state, hermitian_operator):
    return np.real(np.conj(state).dot(hermitian_operator.dot(state)))*grid_size

def cal_energy(state, Hamiltonian):
    return np.real(np.conj(state).dot(Hamiltonian.dot(state)))*grid_size

def normalize(state, prob=None):
    # norms of position components sum to one does not mean a proper dot product computation, 
    # because integration on space involves a size of dx, which is grid_size dependent,
    # or number of \psi position components dependent as the number of sampling.
    # Here we take that integration involves a dx multiplier, an the space integration should be 1 as the normalization.
    if type(prob) == type(None): p = linalg.norm(state)
    else: p=sqrt(np.sum(prob))
    return state / (p*sqrt(grid_size))

def Gaussian_packet(wavelength, mean, std):
    return np.exp(2.j*pi*(x-mean)/wavelength)*np.exp(-(x-mean)*(x-mean)/(4*std*std))/sqrt(sqrt(2*pi)*std)


################################## start learning setting, including some other settings not present in the commandline arguments

half_period_steps = args.time_steps
time_step = 1 / half_period_steps

controls_per_unit_time = args.n_con
assert half_period_steps % controls_per_unit_time == 0, 'We require that time_steps {} to be fully divided by the number of control steps n_con {}.'.format(args.time_steps, args.n_con)
control_interval = round(half_period_steps / controls_per_unit_time)

num_of_episodes = args.num_of_episodes
reward_multiply = args.reward_scale_up
failing_reward = -(args.energy_cutoff)*reward_multiply

t_max = 100.
num_of_saves = args.num_of_saves


if args.input == 'xp':
    data_size = round((2+args.input_moment_order+1)*args.input_moment_order/2)
elif args.input == 'wavefunction':
    data_size = 2*(x_n-10*2)


# we do not plot when we do parallelized computation

#import plot
#plot.set_parameters(x=x, x_max=x_max, dt=time_step, num_of_episodes=num_of_episodes, probability=probability, 
#    reward_multiply=reward_multiply, read_length=read_length, controls_per_unit_time=controls_per_unit_time)


# set the reinforcement learning settings
if __name__ == '__main__':
    import RL
    RL.set_parameters(control_interval=control_interval, t_max=t_max, F_max=args.F_max, failing_reward=failing_reward)

################################## end learning setting


# Below is the worker function for subprocesses, which carries out the control simulations and pushes the experiences and records to queues that are collected and handled by other processes. (Quantum simulation is implemented in a compiled C module)
# Because too many processes using CUDA will occupy a huge amount of GPU memory, we avoid using CUDA in these workers. Instead, these workers ask a manager process when they want to evaluate the neural network, and only the manager process is allowed to use CUDA to evaluate the neural network for the controls.
def Control(net, pipes, shared_buffer, seed, idx):
    simulation = __import__('simulation')
    # seeding
    random = np.random.RandomState(seed)
    simulation.set_seed(random.randint(0,2**31 - 1))
    # preparing pipes
    MemoryQueue, ResultsQueue, ActionPipe, EndEvent, PauseEvent = pipes
    state_data_to_manager = np.frombuffer(shared_buffer,dtype='float32')
    # data input for the neural network
    def get_data_xp(state):
        data = np.empty((data_size,))
        simulation.get_moments(state, data)
        return data.astype(np.float32) 
    def get_data_wavefunction(state):
        return np.hstack((np.real(state[10:-10]),np.imag(state[10:-10]))).astype(np.float32) 
    if args.input == 'xp':
        get_data = get_data_xp    # get_data is the function that returns the input data for AI given a quantum state
    elif args.input == 'wavefunction':
        get_data = get_data_wavefunction

    # random action decision hyperparameters
    EPS_START = 0.2
    EPS_END = 0.004
    EPS_DECAY = args.n_con*t_max*80
    # initialization
    steps_done = 0
    no_action_choice = net.num_of_control_resolution_oneside
    if __name__ != 'controllers' and args.control_strategy!='DQN':
        import controllers 
    def call_force(data):
        nonlocal steps_done
        # apply an \epsilon-greedy strategy:
        eps_threshold = (EPS_START-EPS_END) * exp(-1. * steps_done / EPS_DECAY)
        eps_threshold += EPS_END
        steps_done += args.num_of_actors # this approximates the total steps_done of all the actors
        if random.uniform() < eps_threshold and not args.test:
            last_action=random.randint(no_action_choice*2+1) 
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
    def analytic_controls(state):
        if args.control_strategy=='damping': F = controllers.steepest_descent(state, damping = args.con_parameter)
        elif args.control_strategy=='LQG': F = controllers.LinearQuadratic(state, k = lambda_*args.con_parameter)
        elif args.control_strategy=='semiclassical': F = controllers.Gaussian_approx(state)
        force_max = net.convert_to_force(2*no_action_choice)
        force = F / pi
        force = min(force, force_max)
        force = max(force, -force_max)
        force = round(force/(force_max/no_action_choice))
        force = float(force*(force_max/no_action_choice))
        return force, round(force/(force_max/no_action_choice))+no_action_choice, False
    def init_state():
        wavenumber=random.uniform(-0.3,0.3)
        if wavenumber == 0.: wavelength = float('inf')
        else: wavelength = 1./wavenumber
        state = Gaussian_packet(wavelength=wavelength, mean=0., std=1.)
        t = 0.
        init_time = random.uniform(15.,20.) # initialization time
        while t < init_time: 
            q, x_mean, Fail = simulation.step(state, time_step, 0., gamma)
            t+=time_step
        return state, Fail
    init_energy_cutoff = 7.5
    # do one episode
    def do_episode():
        t = 0.
        # prepare the quantum state
        state, init_Fail = init_state()
        energy = cal_energy(state, Hamil)
        # we retry until we get an initial state with a moderately small energy, i.e. smaller than 0.9 * args.energy_cutoff
        while energy>=init_energy_cutoff or init_Fail:
            state, init_Fail = init_state()
            energy = cal_energy(state, Hamil)
        # force is the parameter before -\pi\hat{x}, which is the physical force divided by \pi
        force = 0.
        last_action = no_action_choice
        # start the simulation loop
        i = 0
        experience = []
        accu_energy = 0.; accu_counter = 0; to_stop = False
        energy_cutoff = args.energy_cutoff if args.train else args.test_energy_cutoff
        while not t >= t_max-0.01*time_step:
            if i % control_interval == 0:
                energy = cal_energy(state, Hamil)
                data = get_data(state)*args.input_scaling
                if energy < energy_cutoff and not to_stop:
                    if args.train and i != 0:
                        experience.append(np.hstack(( last_data, data, 
                            np.array([last_action],dtype=np.float32),  
                            np.array([-energy*reward_multiply],dtype=np.float32) )) )
                else:
                    break
                    # We tried applying an "endpoint" Q value for the AI to learn when it fails. 
                    # However, the strategy would significantly deteriorate the AI's final performance

                if t>50-0.01*time_step: accu_energy += energy; accu_counter += 1
                if args.control_strategy=='DQN':
                    force, last_action, rnd = call_force(data) 
                else: force, last_action, rnd = analytic_controls(state)
                last_data = data
            q, x_mean, Fail = simulation.step(state, time_step, force, gamma)
            i += 1
            if Fail and not to_stop: to_stop = True
            t += time_step
        # push experience into the main process and push results to the manager
        if t>= t_max-0.01*time_step: t=t_max
        avg_energy = accu_energy/accu_counter if t==t_max else energy_cutoff
        if not EndEvent.is_set():
            MemoryQueue.put( (experience, t, avg_energy, to_stop) )
            ResultsQueue.put((t, avg_energy))
        return avg_energy
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
    if args.write_training_data and args.train:
        if os.path.isfile(args.folder_name + '.txt'): os.remove(args.folder_name + '.txt')
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
        worker_data.append(torch.from_numpy(np_memory))
        seed = random.randrange(0,2**31 - 1)
        processes.append( fork.Process( target=Control, args=(copy.deepcopy(net).cpu(), (MemoryQueue, results_queue, conn2, EndEvent, PauseEvent), shared_buffer, seed, n) ) )
    net=net.cuda()
    net.eval()
    # prepare to save
    good_actors=[(args.energy_cutoff/2.,0.) for i in range(num_of_saves)]
    simulated_T = 0.
    performances = []
    episode_passed = 0
    # when receiving a net, check whether the previous net should be stored
    def receive_net():
        if args.test or args.control_strategy!='DQN':
            if not results_queue.empty():
                while not results_queue.empty():
                    result = results_queue.get()
                    performances.append(result[1])
                    with open(os.path.join(args.folder_name, others+'_record.txt'),'a') as f:
                        f.write('{}\n'.format(result[1]))
            return

        if ActorPipe.poll():
            nonlocal net
            if not results_queue.empty():
                nonlocal episode_passed, simulated_T
                while not results_queue.empty():
                    result = results_queue.get()
                    simulated_T += result[0]/2.
                    performances.append(result[1]) # get the avg_energy in the result tuple
                    episode_passed += 1
                    if args.write_training_data:
                        with open(args.folder_name + '.txt','a') as f:
                            f.write('{}, {}\n'.format(simulated_T, result[1]))
                # only if 50 additional episodes have passed, do we consider saving the next model
                if episode_passed > 50 and len(performances) >= 8: 
                    new_avg_energy = np.mean(np.array(performances[-8:]))
                    if new_avg_energy < good_actors[-1][0]:
                        good_actors[-1] = (new_avg_energy, copy.deepcopy(net).cpu())
                        good_actors.sort(key=lambda p: p[0], reverse=False) # sort in increasing order; "reverse" is False, in fact unnecessary
                        print(colored('new avg energy record: {:.5f}'.format(new_avg_energy), 'green',attrs=['bold']))
                        for idx, actor in enumerate(good_actors):
                            if type(actor[1]) != float:
                                torch.save(actor[1].state_dict(), os.path.join(args.folder_name,'{}.pth'.format(idx+1)))
                                existing_record_name = os.path.join(args.folder_name,'{}_record.txt'.format(idx+1))
                                if os.path.isfile(existing_record_name): os.remove(existing_record_name)
                        episode_passed = 0
            performances.clear()
            net.load_state_dict(ActorPipe.recv())
            while ActorPipe.poll():
                net.load_state_dict(ActorPipe.recv())
            net = net.cuda()
            net.eval()
            if args.show_actor_recv: print(colored('new model received', 'yellow'))

    network_input = torch.empty((num_of_processes,data_size), device='cuda')

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
                    data_received_id.append(i) # the data sent through pipe is exactly the id
                    network_input[num_of_data]=worker_data[i][:]
                    num_of_data += 1
                else: process_ended += 1

                while message_conn[i].poll(): # For safety; when False, this while loop is ignored
                    message_conn[i].recv()

        # process the received data
        if num_of_data == 0:
            time.sleep(0.0005) # if no data, wait for 0.5 milisecond
            if args.control_strategy!='DQN': time.sleep(1.)
        else:
            action_values, avg_value, _noise = net(network_input[:num_of_data])
            actions = action_values.max(1)[1].cpu()
            for i,idx in enumerate(data_received_id):
                message_conn[idx].send(actions[i].item())
        # update the network
        receive_net()
    # end, if have left the while loop
    if args.test or args.control_strategy!='DQN':
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
                        if updates_done >= 200.*downscaling_of_default_num_updates or self.pause_event.is_set():
                            with self.pending_training_updates.get_lock():
                                self.pending_training_updates.value -= updates_done
                                updates_done = 0.
                if remaining_updates < 50. *downscaling_of_default_num_updates and self.pause_event.is_set():
                    self.pause_event.clear()
                if self.t_done.value >= self.actor_update_time:
                    self.scale_up_actor_update_time(self.last_achieved_time.value)
                    if not self.ActorReceivePipe.poll() and started and args.control_strategy=='DQN':
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
            return
        def scale_up_actor_update_time(self, achieved_time):
            changed = False
            if achieved_time>80. and self.actor_update_time<=150.:
                self.actor_update_time = 800.; changed = True
            elif achieved_time>10. and self.actor_update_time<=25.:
                self.actor_update_time = 50.; changed = True
            elif achieved_time>5. and self.actor_update_time<=10.:
                self.actor_update_time = 25.; changed = True
            if changed and args.train: print('actor_update_time adjusted to {:.1f}'.format(self.actor_update_time))
        def adjust_learning_rate(self):
            if self.train.backup_period != self.backup_period and self.learning_in_progress_event.is_set():
                self.train.backup_period = self.backup_period
                if 1.e-3 < args.lr:
                    args.lr = min(1.e-3, args.lr)
                    if args.train:
                        for param_group in self.train.optim.param_groups: param_group['lr'] = args.lr
                        print(colored('learning rate set to {:.2g}'.format(args.lr),attrs=['bold']))
            # the learning rate schedule is written in "arguments.py"
            if self.episode.value > args.lr_schedule[self.lr_step][0] and self.last_achieved_time.value == t_max:
                if args.lr_schedule[self.lr_step][1] < args.lr:
                    args.lr = min(args.lr_schedule[self.lr_step][1], args.lr)
                    if args.train:
                        for param_group in self.train.optim.param_groups: param_group['lr'] = args.lr
                        print(colored('learning rate set to {:.2g}'.format(args.lr),attrs=['bold']))
                self.lr_step += 1



    # system settings, checks and the framework
    def check_C_module_and_compile():
        if not args.compile:
            try:
                simulation = __import__('simulation')
                (c_x_n, c_grid_size, c_lambda, c_mass, c_moment) = simulation.check_settings()
                default_str = ' of the existing C module ({}) does not match the current task ({}). Recompile.\n'
                if c_x_n != x_n:
                    print(colored(('X_N'+default_str).format(c_x_n, x_n), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_grid_size != grid_size:
                    print(colored(('Grid_size'+default_str).format(c_grid_size, grid_size), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_lambda != lambda_:
                    print(colored(('\lambda'+default_str).format(c_lambda, lambda_), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_mass != mass:
                    print(colored(('Mass'+default_str).format(c_mass, mass), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_moment != args.input_moment_order:
                    print(colored(('Input distribution moment order'+default_str).format(c_moment, args.input_moment_order), 'yellow',attrs=['bold']))
                    args.compile = True
            except (ModuleNotFoundError, AttributeError) as e:
                args.compile = True
            if args.compile: time.sleep(1)
        if args.compile:
            code = os.system('python{} setupC.py --x_max {} --grid_size {} --lambda {} --mass {} --moment {}'.format(sys.version[:3],
                           x_max,grid_size,lambda_,mass,args.input_moment_order))
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
    capacity = round(args.size_of_replay_memory*controls_per_unit_time*t_max) if args.train else 1
    memory = RL.Memory(capacity = capacity, data_size = data_size * 2 + 2, policy = 'random', passes_before_random = 0.2)
    # define the neural network
    net = RL.direct_DQN(data_size).cuda()
    # set the task
    if args.train or args.control_strategy!='DQN':
        train = RL.TrainDQN(net, memory, batch_size = args.batch_size, gamma=0.99, backup_period = args.target_network_update_interval, args=args)
        del net
        # the main function of training
        if args.train: 
            main = Main_System(train, num_of_processes=args.num_of_actors)
            main(num_of_episodes)
        # when we do not train and we test the result of analytic strategies
        elif args.control_strategy!='DQN': 
            string = args.control_strategy if args.control_strategy=='semiclassical' else args.control_strategy+str(args.con_parameter)
            main = Main_System(train, num_of_processes=args.num_of_actors, others=string)
            main(args.num_of_test_episodes)
    # if we test existing models, we use a loop to iterate over the models
    else:
        # find all models to test that end with no extension or '.pth' in the given directory
        import glob
        test_nets = []
        for name in glob.glob(os.path.join(args.folder_name,'*')):
            file_name, ext = os.path.splitext(os.path.basename(name))
            if (ext=='.pth' or ext=='') and os.path.isfile(name): test_nets.append((file_name, torch.load(name)))
        assert len(test_nets)!=0, 'No model found to test'
        from utilities import isfloat, isint
        test_nets = sorted([t for t in test_nets if isfloat(t[0])], key = lambda t: float(t[0])) + sorted([t for t in test_nets if not isfloat(t[0])])
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
