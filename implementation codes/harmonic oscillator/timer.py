import time

# a simple timer to benchmark different parts of the code
# "timer(idx)" is called when one wants to add the time elapsed since the last call to the record "idx",
# and when one does print(timer), its internal records are reset.

class Timer:
    times = []
    def __init__(self):
        self.time_record = time.time()
    def __call__(self, i):
        length = len(self.times)
        if length <= i: self.times.extend([0. for j in range(i-length+1)])
        current = time.time()
        diff = current - self.time_record
        self.time_record = current
        self.times[i] += diff
    def __getitem__(self, item):
        return self.times[item]
    def reset(self):
        for i in range(len(self.times)): self.times[i] = 0.
    def start(self):
        self(0)
        self.reset()
    def __repr__(self):
        s = '[ '
        for value in self.times:
            s += '{:.4g} '.format(value)
        self.reset()
        return s +']'

def print_elapsed_time(time_of_start):
    time_of_end = time.time()
    print('\n'+time.ctime())
    time_elapsed = time_of_end - time_of_start
    hours = round(time_elapsed // 3600.)
    minutes = round((time_elapsed - hours*3600.) // 60.)
    seconds = time_elapsed - 3600.*hours - 60.*minutes
    print('spent {} h {} min {:.0f} s'.format(hours, minutes, seconds))
