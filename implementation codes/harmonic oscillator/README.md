When using measurement outcomes as the input, the network receives a large number of input values and it is deeper and much harder to train. As a result, the reinforcement learning process is not as stable as other cases, and it only works well with moderate hyperparameter settings. The settings that generate our results are set to the default, and can be run by ```python main_parallel.py --input measurements```

```python main_parallel.py --input wavefunction --phonon_cutoff 10 --reward_scale_up 1.```
