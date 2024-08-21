import h5py
import pandas as pd
import numpy as np

path = '/mnt/SkyGPT/Data_publication/'
filename = 'test_set_2019nov_dec.hdf5'
read_file = path + filename

with h5py.File(read_file, "r") as f:
    for key in f.keys():
        print(key) #Names of the root level object names in HDF5 file - can be groups or datasets.
        print(type(f[key])) # get the object type: usually group or dataset

    #Get the HDF5 group; key needs to be a group name from above
    group = f[key]

    #Checkout what keys are inside that group.
    for key in group.keys():
        print(key)

        # This assumes group[some_key_inside_the_group] is a dataset, 
        # and returns a np.array:
        data = group[key][()]
        #Do whatever you want with data
        print(f'Extracting {key}')
        np.save(path + key + '.npy', data, allow_pickle=True)

    f.close()
