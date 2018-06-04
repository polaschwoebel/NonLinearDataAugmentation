#%%
import numpy as np
from joblib import Parallel, delayed
from joblib import load, dump
import time
import tempfile
import os
import shutil 

#%%
def sum_row(inputt, output, i):
    sum_ = inputt[i, :].sum()
    output[i] = sum_


rng = np.random.RandomState(42)
samples_name = "samples"
sums_name = "sums"
# Generate some data and an allocate an output buffer
samples = rng.normal(size=(10, int(1e6)))

# Pre-allocate a writeable shared memory map as a container for the
# results of the parallel computation
sums = np.memmap(sums_name, dtype=samples.dtype,
                 shape=samples.shape[0], mode='w+')

#%%
# Dump the input data to disk to free the memory
dump(samples, samples_name)

# Release the reference on the original in memory array and replace it
# by a reference to the memmap array so that the garbage collector can
# release the memory before forking. gc.collect() is internally called
# in Parallel just before forking.
samples = load(samples_name, mmap_mode='r')

# Fork the worker processes to perform computation concurrently
res = np.zeros(samples.shape[0])
start = time.time()
[sum_row(samples, res, i) for i in range(samples.shape[0])]
print("SEQ time: ", (time.time() - start) / 60)

start = time.time()
Parallel(n_jobs=-1, backend="threading")(delayed(sum_row)(samples, sums, i)
                   for i in range(samples.shape[0]))

print("PAR time: ", (time.time() - start) / 60)
    
shutil.rmtree(folder)
