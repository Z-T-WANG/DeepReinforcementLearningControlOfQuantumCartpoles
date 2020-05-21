Our implementation codes are written in Python with customized C++ extensions, which make use of the Intel MKL library to calculate the quantum simulation. To run the codes, it is required that Pytorch and Numba are installed in the Python environment, and that the system environment variable ```${MKLROOT}``` is set correctly.
We also assume that the GPU based computation, i.e. CUDA, is available. 

In each of the folders above, the main program is the python script ```main_parallel.py```, which can be run as ```python3.x main_parallel.py```. The arguments and settings are organized in file ```arguments.py``` and can be invoked by using commandline arguments.

### Code Organization
Our implementation basically includes three different submodules. The first is the quantum simulation submodule, which is coded in ```simulation.cpp``` and is compiled for Python by invoking the script ```setupC.py```. The second one is the deep learning submodule, which includes the reinforcement learning algorithm, prioritized sampling and the neural networks, which are coded in ```layers.py``` ```optimizer.py``` and ```RL.py```. The last submodule is the manager of the whole system, which controls the workers that carry out the quantum simulation and controls the reinforcement learning progress, which is coded in ```main_parallel.py```. 
The script ```main_parallel.py``` first receives and processes the arguments defined in ```arguments.py``` and checks the C++ quantum simulation module, and then it allocates memory and starts parallelized training/testing.
