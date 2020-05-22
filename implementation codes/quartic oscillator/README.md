As the task is highly non-linear, we find that the training is highly initialization-dependent when it begins, and sometimes the same setting of training does not work for different random seeds. Especially, if the training loss does not decrease rapidly at the beginning, it may cause an avalanche effect and the AI may not start its learning progress. Therefore, the training progress is dependent on the initial learning progress and the initial learning rate, which also slightly affects its final performance. Nevertheless, we find that our default training setting works for most of the different random seeds. If the training fails, one may try running the code once more or change to a different random seed.

For reference, the commands we have used to produce our results are listed below. However, due to the asynchronous nature of parallelization, it is impossible to exactly reproduce our results, but we believe that similar results can still be obtained easily.\
```python3 main_parallel.py --input xp --input_scaling 0.1 --init_lr 2.e-3 --seed 1```\
```python3 main_parallel.py --input wavefunction --init_lr 1.e-3 --seed 0　```

To test the trained models, simple add argument ```--test```.

The control strategies of the comparison group are tested by\
```python3 main_parallel.py --control_strategy damping --con_parameter 0.4```\
```python3 main_parallel.py --control_strategy LQG --con_parameter 2.8```\
```python3 main_parallel.py --control_strategy semiclassical```
