Our implementation codes are written in Python with customized C++ extensions that make use of Intel MKL library to do the quantum simulation.
To run the codes, it is required that Pytorch and Numba are installed in the Python environment, and that the environment variable ```${MKLROOT}``` is set.
We also assume that GPU based computation, i.e. CUDA, is available. The main program is the file ```main_parallel.py```, and the settings are in
```arguments.py```.
