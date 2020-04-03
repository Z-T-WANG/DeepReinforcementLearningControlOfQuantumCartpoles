import argparse, math

parser = argparse.ArgumentParser()

# setting of quantum simulation (compiled in a C module which is importable by Python)
parser.add_argument('--lambda', default = -0.01 , type=float, metavar='\lambda',
                    help='the strength of the quartic anharmonic oscillator in units of \pi * (m_c^2*\omega_c^3 / \hbar). Note that the \lambda we use is this argument multiplied by \pi. The potential can be simply regarded as unitless \lambda*x^4 with \hbar=1.')
parser.add_argument('--mass', default = 1., type=float, metavar='m',
                    help='the mass of the simulated particle in units of m_c / \pi. Note that the mass we use is this argument divided by \pi.')
parser.add_argument('--x_max', default=13., type=float, metavar='x_{max}',
                    help='the distance from the center to the border of the simulation space')
parser.add_argument('--grid_size', default=0.05, type=float, metavar='h',
                    help='the number of discretization points of the simulated space')
parser.add_argument('--gamma', default = 1. , type=float, metavar='\gamma',
                    help='the measurement strength \gamma on the particle multiplied by a factor of \pi')
parser.add_argument('--time_steps', default = 360*8, type=int,
                    help="the number of time steps for simulating time 1. The default is 360*8. Usually we don't need to change it.")
parser.add_argument('--n_con', default = 18, type=int, metavar='n_{con}',
                    help=r'the number of control steps per time 1 or \frac{1}{\omega_c}. For simplicity, it is required to divide the number of time steps per time 1.')
parser.add_argument('--F_max', default = 5., type=float, metavar='F_{max}',
                    help=r'the maximum control force allowed, multiplied by a factor of \pi, i.e. in units of \pi\sqrt{\hbar m_c \omega_c^3}.')
parser.add_argument('-c','--compile', action='store_true',
                    help='whether to force a compilation if a existing file exist')

# where to store models, whether to test models or LQG control
parser.add_argument('--save_dir', default='', type=str,
                    help='the directory to save trained models. It defaults to a conventional naming style that is "inputType_lambda_gamma".')
parser.add_argument('--control_strategy', default = 'DQN', choices=['DQN', 'damping', 'LQG', 'semiclassical'],
                    help='the control strategy to use to compare the different performances')
parser.add_argument('--con_parameter', default = 1.9, type=float,
                    help='an additional undetermined control parameter for applying the damping or LQG control')
parser.add_argument('--test', action='store_true',
                    help='whether to test existing trained models rather than to train')
parser.add_argument('--load_dir', default='', type=str,
                    help='the directory of models to test')
parser.add_argument('--num_of_test_episodes', default=4000, type=int,
                    help='the number of episodes to test and collect performance data for each model')

# training settings
parser.add_argument('--input', default = 'xp', choices=['xp', 'wavefunction'],
                    help='the input to the neural network. It can be "xp" moments or the "wavefunction".')
parser.add_argument('--input_moment_order', default = 5, type=int,
                    help='the order of the distribution moments that are used as input to the neural network (default 5)')
parser.add_argument('--batch_size', default = 512, type=int,
                    help='the sampled minibatch size per update step in training')
parser.add_argument('--n_times_per_sample', default = 8, type=int,
                    help='the number of times each experience is sampled and learned')
parser.add_argument('--size_of_replay_memory', default = 8000, type=int,
                    help='the size of the replay memory that stores the accumulated experiences, in units of full-length time 100 episodes. Its default value is 5000.')
parser.add_argument('--target_network_update_interval', default = 300, type=int,
                    help='the number of performed gradient descent steps before updating the target network. \nThe target network is a lazy copy of the currently trained network, i.e., it is updated to the current network only after sufficiently many gradient descent steps are done. It is used in DQN training to provide a more stable evaluation of the current Q value. The number of the gradient descent steps is this "target_network_update_interval" parameter.')
parser.add_argument('--train_episodes_multiplicative', default = 3., type=float,
                    help=r'the multiplicative factor that rescales the default number of simulated episodes (20000), counted only when the learning is in progress. The counting of episodes will be reset to 1 when the controller achieves an average time of 20, and it will be reset again if the controller drops to an average time of 5. This rescaling factor also rescales the learning rate schedule, the save_interval, and the maximum_trails_before_giveup.')
parser.add_argument('--maximum_trails_before_giveup', default = 40000, type=int,
                    help=r'the maximal number of simulated episodes when the learning does not proceed. If the simulated episodes exceed this value, we give up training.')
parser.add_argument('--init_lr', default = 2e-3, type=float,
                    help='the initial learning rate. The learning rate will be decayed to 4e-5 at episode 1000, 8e-6 at 3000, 2e-6 at 5000, 4e-7 at 6500 and 1e-7 at 8000 when the current learning rate is higher.')
parser.add_argument('--alive_reward', default = 10., type=float,
                    help='the reward of not falling for the AI')
parser.add_argument('--input_scaling', default = 1., type=float,
                    help='a multiplicative factor of the input data to the AI. This is to avoid a possibly different scale between the input and the output of the AI, which may require too many unnecessary update steps during learning. This feature is set to 1 and thus disabled by default.')
parser.add_argument('--num_of_actors', default = 40, type=int,
                    help='the number of actors, i.e. the number of working processes that repeatedly do the control to accumulate experiences.')
parser.add_argument('--show_actor_recv', action='store_true',
                    help='to signify when a new model is received by the actors during training')
parser.add_argument('--save_interval', default = 200, type=int,
                    help='the number of episodes passed before we save the next model. It defaults to one save per 200 episodes. Note that this value will be rescaled by "train_episodes_multiplicative".')
parser.add_argument('--num_of_saves', default = 20, type=int,
                    help='the number of models to save. Later models overwrite earlier ones.')
parser.add_argument('--write_training_data', action = 'store_true',
                    help='whether to store the data that are used to plot training curves')

# system config
parser.add_argument('--gpu_id', default = 0, type=int,
                    help='the index of the GPU to use')
parser.add_argument('--seed', default = -1, type=int,
                    help='the random seed. When not set, it is random by default.')

parser.add_argument('--eta', default = 1., type=float,
                    help='the measurement efficiency. \eta being smaller than 1 means that there are additional measurements performed by the environment but are ignored.')

args = parser.parse_args()

assert 0. < args.eta <= 1., "The measurement efficiency should be larger than 0 and equal to or smaller than 1. It is currently {:.3g}.".format(args.eta)

num_of_discrete = math.floor(args.x_max/args.grid_size+1e-4)
args.x_n = num_of_discrete*2+1
args.x_max = num_of_discrete*args.grid_size

args.num_of_episodes = round(20000*args.train_episodes_multiplicative)
args.maximum_trails_before_giveup = round(args.maximum_trails_before_giveup*args.train_episodes_multiplicative)
args.lr_schedule = [(round(t[0]*args.train_episodes_multiplicative) if t[0]!=float('inf') else t[0], t[1]) for t in [(4000,8e-5), (9000,2e-5), (14000,4e-6), (16000,8e-7), (18000,2e-7), (20000, 0.)]]
args.save_interval=round(args.save_interval*args.train_episodes_multiplicative)
# set the default learning rate
args.lr = args.init_lr

# decide whether to train based on the commandline arguments
if args.test: args.train = False
elif args.control_strategy!='DQN': args.train = False
else: args.train = True

assert args.__dict__['lambda']<0, "The \lambda coefficient should be negative to define the quartic cartpole."

# prepare the path name
args.folder_name = '{}Input_lm{}_ga{}'.format(args.input, args.__dict__['lambda'], args.gamma) if args.save_dir == '' else args.save_dir
if args.test:
    if args.load_dir != '':
        args.folder_name = args.load_dir # "load_dir" overrides "save_dir"
    else: 
        args.load_dir = args.folder_name # if "save_dir" is provided but "load_dir" is not, we assume that "load_dir" is the same as "save_dir"
elif args.load_dir != '' and args.save_dir == '': args.folder_name = args.load_dir

