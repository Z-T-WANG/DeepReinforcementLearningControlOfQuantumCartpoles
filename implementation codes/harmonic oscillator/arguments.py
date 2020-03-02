import argparse

parser = argparse.ArgumentParser()

# setting of quantum simulation (compiled in a C module which is importable by Python)
parser.add_argument('--omega', default = 1. , type=float, metavar='\omega',
                    help='the angular frequency of the harmonic oscillator (in units of \pi*time^{-1} or \pi*\omega_c). This option makes no effect and it is always set to be the default, because the rescaling of \omega only amounts to rescaing of the time and space and is trivial.')
parser.add_argument('--n_max', default = 70, type=int, metavar='n_{max}',
                    help='the highest energy level of simulation. The last output (Failure) will be True if the amplitude on this level is too high.')
parser.add_argument('--gamma', default = 1. , type=float, metavar='\gamma',
                    help='the measurement strength \gamma on the particle multiplied by a factor of \pi')
parser.add_argument('--n_con', default = 18, type=int, metavar='n_{con}',
                    help=r'the number of control steps per time 1 or \frac{1}{\omega_c}. For simplicity, it is required to divide the number of time steps per time 1.')
parser.add_argument('--F_max', default = 5., type=float, metavar='F_{max}',
                    help=r'the maximum control force allowed, multiplied by a factor of \omega, i.e. in units of \omega\sqrt{\hbar m_c \omega_c}. With the default \hbar = m_c * \omega_c = m * \omega = 1 and \omega = k = \pi, F_max is exactly the maximum shift of the potential when applying the force.')
parser.add_argument('--phonon_cutoff', default = 20., type=float,
                    help=r'the maximum average phonon number we allow during simulation, beyond which we will stop the simulation so as to avoid high energy samples to be learned by the AI. This is because when the learning is difficult and when the AI learns high energy samples (associated with high loss) and low energy samples simutaneously, it attempts to fit into the high energy samples more and performs worse on low energy situations, and then the accumulated experience will always contain high energy samples and the AI will achieve a bad performance in the end. To avoid this, we impose an upper bound on its learned energy region. This strategy stabilizes the learning and makes it faster.')
parser.add_argument('-c','--compile', action='store_true',
                    help='whether to force a compilation if a existing file exist')

# where to store models, whether to test models or LQG control
parser.add_argument('--save_dir', default='', type=str,
                    help='the directory to save trained models. It defaults to a conventional naming style that is "inputType_omega_gamma".')
parser.add_argument('--LQG', action='store_true',
                    help='whether to use the LQG control without training')
parser.add_argument('--train_with_LQG', action='store_true',
                    help='when LQG control is used, whether to train the model on the experience obtained by LQG control')
parser.add_argument('--test', action='store_true',
                    help='whether to test existing trained models rather than to train')
parser.add_argument('--load_dir', default='', type=str,
                    help='the directory of models to test')
parser.add_argument('--num_of_test_episodes', default=500, type=int,
                    help='the number of episodes to test and collect performance data for each model')

# training settings
parser.add_argument('--input', default = 'xp', choices=['xp', 'wavefunction', 'measurements'],
                    help='the input to the neural network. It can be "xp" moments, the "wavefunction", or the "measurements".')
parser.add_argument('--batch_size', default = 512, type=int,
                    help='the sampled minibatch size per update step in training')
parser.add_argument('--n_times_per_sample', default = 8, type=int,
                    help='the number of times each experience is sampled and learned')
parser.add_argument('--size_of_replay_memory', default = 0, type=int,
                    help='the size of the replay memory that stores the accumulated experiences, in units of full-length episodes.\nIts default value for "xp" and "wavefunction" input is 5000, and is 1000 for "measurements" input. When this argument receives a non-zero value, the default is overridden.')
parser.add_argument('--target_network_update_interval', default = 300, type=int,
                    help='the number of performed gradient descent steps before updating the target network. \nThe target network is a lazy copy of the currently trained network, i.e., it is updated to the current network only after sufficiently many gradient descent steps are done. It is used in DQN training to provide a more stable evaluation of the current Q value. The number of the gradient descent steps is this "target_network_update_interval" parameter.')
parser.add_argument('--num_of_episodes', default = 9000, type=int,
                    help=r'the number of simulated episodes, each of time 100, i.e. \frac{100}{\omega_c}. The counting of episodes will be reset to 1 when the controller achieves time 100 for the first time, so it corresponds to the number of episodes after learning has started.')
parser.add_argument('--init_lr', default = 4e-4, type=float,
                    help='the initial learning rate. The learning rate will be decayed to 4e-5 at episode 1000, 8e-6 at 3000, 2e-6 at 5000, and 4e-7 at 7000 when the current learning rate is higher.')
parser.add_argument('--phonon_offset', default = 0.3, type=float,
                    help='offset (decrease) the phonon number by a constant amount when calculating the reward. By doing so, the Q value learned by the networks can become closer to zero.')
parser.add_argument('--reward_scale_up', default = 10., type=float,
                    help='a multiplicative factor of the reward for the AI')
parser.add_argument('--input_scaling', default = 1., type=float,
                    help='a multiplicative factor of the input data to the AI. This is to avoid a possibly different scale between the input and the output of the AI, which may require too many unnecessary update steps during learning. This feature is set to 1 and thus disabled by default.')
parser.add_argument('--num_of_actors', default = 10, type=int,
                    help='the number of actors, i.e. the number of working processes that repeatedly do the control to accumulate experiences.')
parser.add_argument('--show_actor_recv', action='store_true',
                    help='to signify when a new model is received by the actors during training')
parser.add_argument('--write_training_data', action = 'store_true',
                    help='whether to output the data that are used to plot training curves to a file')

# system config
parser.add_argument('--gpu_id', default = 0, type=int,
                    help='the index of the GPU to use')
parser.add_argument('--seed', default = -1, type=int,
                    help='the random seed. When not set, it is random by default.')

args = parser.parse_args()

# set the default size of replay memory
if args.size_of_replay_memory==0:
    if args.input == 'xp': args.size_of_replay_memory = 5000
    elif args.input == 'wavefunction': args.size_of_replay_memory = 5000
    elif args.input == 'measurements': args.size_of_replay_memory = 1000 # its memory consumption is large and we haven't optimized specifically for it; 
                                                                        # if we optimize, many assumtions and strategies of its memory management
                                                                        # need to be different and inconsistent with the cases of other inputs
# set the default learning rate
args.lr = args.init_lr

# decide whether to train based on the commandline arguments
if args.test: args.train = False
elif args.LQG and not args.train_with_LQG: args.train = False
else: args.train = True

# prepare the path name
args.folder_name = '{}Input_om{}_ga{}'.format(args.input, args.omega, args.gamma) if args.save_dir == '' else args.save_dir
if args.test:
    if args.load_dir != '':
        args.folder_name = args.load_dir # "load_dir" overrides "save_dir"
    else: 
        args.load_dir = args.folder_name # if "save_dir" is provided but "load_dir" is not, we assume that "load_dir" is the same as "save_dir"
elif args.load_dir != '' and args.save_dir == '': args.folder_name = args.load_dir

