Our implementation codes are written in Python with customized C++ extensions, which make use of the Intel MKL library to calculate the quantum simulation. To run the codes, it is required that Pytorch and Numba are installed in the Python environment, and that the environment variable ```${MKLROOT}``` is set.
We also assume that the GPU based computation, i.e. CUDA, is available. 

The main program is the file ```main_parallel.py```, and the arguments and settings are organized in
```arguments.py``` and can be invoked by commandline arguments.
