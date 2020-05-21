When using measurement outcomes as the input, the network receives a large number of input values and the network becomes deeper and harder to train. As a result, the reinforcement learning process is not as stable as the other cases, and we find that it only works well when using moderate hyperparameter settings. The settings that generated our results have been set to the default, and can be run by ```python main_parallel.py --input measurements```

```python main_parallel.py --input wavefunction --phonon_cutoff 10 --reward_scale_up 1.```
