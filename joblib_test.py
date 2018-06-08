##%%
#import numpy as np
#from joblib import Parallel, delayed
#from joblib import load, dump
#import time
#import tempfile
#import os
#import shutil 

#%%
import numpy as np
from scipy import sparse
import time
import memory_profiler


#%%
def blowup1(S):
    (m, n) = S.shape
    S2 = sparse.lil_matrix((3*m,3*n))
    S2[0::3,0::3] = S
    S2[1::3,1::3] = S
    S2[2::3,2::3] = S
    return S2

def blowup2(S):
    (m, n) = S.shape
    #Sf = S.flatten()
    d = sparse.coo_matrix((3*m,3*n), dtype = np.float32)
    d.data = np.repeat(S.flatten(), 3)
    d.row =  np.repeat(np.arange(3*m).reshape(m,3), n, axis = 0).flatten()
    d.col = np.repeat(np.arange(3*n).reshape(1,-1), m, axis = 0).flatten()
    return d

@profile
def blowup3(S):
    (m, n) = S.shape
    Sf = S.flatten()
    d = sparse.coo_matrix((3*m,3*n), dtype = np.float32)
    d.data = np.repeat(Sf, 3)
    a = np.arange(3*m).reshape(m,3)
    a = np.repeat(a, n, axis = 0)
    a = a.flatten()
    d.row =  a
    d.col = np.repeat(np.arange(3*n).reshape(1,-1), m, axis = 0).flatten()
    return d

#S = np.array([[1,2,3],[4,5,6],[7,8,9],[55,44,33]])
S = np.random.randint(0, 10, (300,300))

#start = time.time()
res1 = blowup3(S).toarray()
#print("time 1 ", time.time() - start)
#start = time.time()
#res2 = blowup3(S).toarray()
#print("time 1 ", time.time() - start)
#print(sum(sum(res1 - res2)))
#%%
a = np.array([[3,2,1],[7,6,5],[1,4,8]])


#%%
#points = np.array([[1,1,1],[1,3,3],[4,6,6],[8,8,8]])
#mask = np.array([[0,1],[0,1]]).astype(np.uint8)
#points[mask.flatten() == 1] = np.array([[66,66,66],[2,2,2]])
#print(points)
##%%
#a = [1,2,3,4,5]
#l = [0.85,0.90]
#s = sorted(a, reverse = False)
#arr = np.array(s)
#for i in l:
#    m = arr[np.diff(np.cumsum(arr) >= sum(arr) * i)]
#
#
##%%
#def sum_row(inputt, output, i):
#    sum_ = inputt[i, :].sum()
#    output[i] = sum_
#
#
#rng = np.random.RandomState(42)
#samples_name = "samples"
#sums_name = "sums"
## Generate some data and an allocate an output buffer
#samples = rng.normal(size=(10, int(1e6)))
#
## Pre-allocate a writeable shared memory map as a container for the
## results of the parallel computation
#sums = np.memmap(sums_name, dtype=samples.dtype,
#                 shape=samples.shape[0], mode='w+')
#
##%%
## Dump the input data to disk to free the memory
#dump(samples, samples_name)
#
## Release the reference on the original in memory array and replace it
## by a reference to the memmap array so that the garbage collector can
## release the memory before forking. gc.collect() is internally called
## in Parallel just before forking.
#samples = load(samples_name, mmap_mode='r')
#
## Fork the worker processes to perform computation concurrently
#res = np.zeros(samples.shape[0])
#start = time.time()
#[sum_row(samples, res, i) for i in range(samples.shape[0])]
#print("SEQ time: ", (time.time() - start) / 60)
#
#start = time.time()
#Parallel(n_jobs=-1, backend="threading")(delayed(sum_row)(samples, sums, i)
#                   for i in range(samples.shape[0]))
#
#print("PAR time: ", (time.time() - start) / 60)
#    
#shutil.rmtree(folder)
