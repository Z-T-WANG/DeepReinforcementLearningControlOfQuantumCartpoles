We find that the training is initialization-dependent and if the training loss does not decrease rapidly at the beginning, the final performance will be worse. Therefore, the initial learning progress and the initial learning rate are important. The commands we have used to produce our results are listed below. However, due to the asynchronous nature of parallelization, it is impossible to exactly reproduce our results, but we believe that similar results can still be obtained easily.\
```python3 main_parallel.py --input xp --init_lr 2.e-3```\
```python3 main_parallel.py --input wavefunction --init_lr 1.e-3 --input_scaling 10.```

To test the trained models, simple add argument ```--test```.

The control strategies of the comparison group are tested by\
```python3 main_parallel.py --control_strategy damping --con_parameter 1.9```\
```python3 main_parallel.py --control_strategy LQG --con_parameter 260```\
```python3 main_parallel.py --control_strategy semiclassical```












