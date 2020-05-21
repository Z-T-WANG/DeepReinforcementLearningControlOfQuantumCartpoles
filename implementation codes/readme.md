Our implementation codes are written in Python with customized C++ extensions, which make use of the Intel MKL library to calculate the quantum simulation. To run the codes, it is required that Pytorch and Numba are installed in the Python environment, and that the environment variable ```${MKLROOT}``` is set.
We also assume that the GPU based computation, i.e. CUDA, is available. 

In each of the folders above, the main program is the python script ```main_parallel.py```, which can be run as ```python3.x main_parallel.py```. The arguments and settings are organized in file ```arguments.py``` and can be invoked by using commandline arguments.
