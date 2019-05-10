
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import os
import time
import numpy as np
import gc

class dummy:
    def __init__(self, x):
        self.x = x
        self.arr = np.random.rand(int(100e6))
        
    def donothing(self):
        print(f"I (obj {id(self)}) am in {os.getpid()}")
        
    def modify(self, f):
        self.x = self.x*f
    
    def calc(self, f):
        return self.x*f

# @profile
def worker_func(obj, f):
    print(mp.current_process())
    print('process id:', os.getpid())
    time.sleep(2)
    y = np.random.rand(int(100e6))
    obj.donothing()
    obj.calc(f)
    obj.donothing()
    print(id(obj), obj.x)
    time.sleep(2)
    obj.modify(f)
    print(id(obj), obj.x)
    obj.donothing()
    time.sleep(2)
    
# @profile
def test2():
    dobj = dummy(10)
    x_s = [4,2,3,5,3,2,1,2]
    w_s = []
    gc.collect()
    l = gc.get_objects()
    for item in l:
        print(type(l))
    for i in range(3):
        w = mp.Process(target=worker_func, args=(dobj, x_s[i]))
        w.start()
        w_s.append(w)
    while any([w.is_alive() for w in w_s]):
        time.sleep(0.1)


def my_func(x, d):
    print(mp.current_process())
    print('process id:', os.getpid())
    print('1:', mp.current_process(), id(d), id(d['y']), id(d['arr']), d['y'], d['arr'][x+500])
    d['y'] = 100
    print('2:', mp.current_process(), id(d), id(d['y']), id(d['arr']), d['y'], d['arr'][x+500])
    d['arr'][x:x+1000] = 10
    print('3:', mp.current_process(), id(d), id(d['y']), id(d['arr']), d['y'], d['arr'][x+500])
    for i in range(20):
        y = np.sum(d['arr'])
    return x*np.sum(d['arr'])

def test():
    arr = np.random.rand(int(1000e6))
    d = {'y': 0, 'arr': arr}
    print(arr.nbytes)
    x_s = [4,2,3,5,3,2,1,2]
    print(x_s, arr)
    w_s = []
    print([arr[x] for x in x_s])
    for i in range(3):
        w = mp.Process(target=my_func, args=(x_s[i], d))
        w.start()
        w_s.append(w)
    while any([w.is_alive() for w in w_s]):
        time.sleep(0.1)
    print([arr[x] for x in x_s])
    
    
    
#     with mp.Pool(2) as pool:
# #     result = pool.map(my_func, [4,2,3,5,3,2,1,2])
# #     result_set_2 = pool.map(my_func, [4,6,5,4,6,3,23,4,6])
# #     print(result)
# #     print(result_set_2)
#         results = [pool.apply_async(my_func, (x, arr)) for x in x_s]
#         results = [r.get() for r in results]
#         print(results, arr)

if __name__ == '__main__':
    test2()
