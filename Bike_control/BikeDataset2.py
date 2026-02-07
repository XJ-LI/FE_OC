from typing import Tuple, Union
import os
import urllib.request
import tqdm
from scipy.io import loadmat

import torch
import numpy as np

from FunctionEncoder.Dataset.BaseDataset import BaseDataset

'''
Dataset for 4D Bike OC problem
544 different obstacle config for training
100 trajectories randomly solved for each problem setting
used for training function encoder
'''

pbar, count = None, 0
desc = ""

def show_progress(block_num, block_size, total_size):
    global pbar
    global count
    global desc
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded - count)
        count = downloaded
    else:
        pbar.close()
        pbar = None
        count = 0

os.makedirs("./Dataset", exist_ok=True)

if not os.path.exists("./Dataset/bicycle_dataset_2_shuffled.mat"):
    ln = "https://drive.usercontent.google.com/download?id=1gjUW8XC7hNGMjS5P-QYvjTpIh0N50zbR&export=download&authuser=0&confirm=t&uuid=bd5de6a0-3c39-46de-a8c3-5b6b50d6aa05&at=APcXIO1WnaRE3F9RhZF20opWcrEq%3A1769314040973"
    desc = "Bike 2 Obstacle OC Problem"
    urllib.request.urlretrieve(ln, "./Dataset/bicycle_dataset_2_shuffled.mat", reporthook=show_progress)
print("Data set located.\n\n")

assert os.path.exists('./Dataset/bicycle_dataset_2_shuffled.mat'), "bicycle_dataset_2_shuffled.mat not found"


dataset = loadmat('./Dataset/bicycle_dataset_2_shuffled.mat')

print(f"There are {dataset['state_time'].shape[0]} examples in the training dataset, each corresponding to a different obstacle configuration")
print(f"For each obstacle config we generate {dataset['state_time'].shape[1]} different trajectories over {dataset['state_time'].shape[2]} time steps between 0 and 5 ")
print(f"Problem has dimension {dataset['state_time'].shape[3]}")

class BikeDataset(BaseDataset):
    def __init__(self,
                 test = False, 
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 n_functions:int=None,
                 n_examples:int=None,
                 n_queries:int=None,
                 # deprecated arguments
                 n_functions_per_sample:int = None,
                 n_examples_per_sample:int = None,
                 n_points_per_sample:int = None,
                 ):
        # default arguments. These default arguments will be placed in the constructor when the arguments are deprecated. but for now they live here.         
        if n_functions is None and n_functions_per_sample is None:
            n_functions = 16
        if n_examples is None and n_examples_per_sample is None:
            n_examples  = 1000
        if n_queries is None and n_points_per_sample is None:
            n_queries   = 2000         
        
        super().__init__(input_size=(5,),
                         output_size=(2,),
                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         # deprecated arguments
                         total_n_functions=None,
                         total_n_samples_per_function=None,
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         )     
        self.test = test
        if self.test:
            inputs  = torch.tensor(dataset['state_time'], device=self.device, dtype=self.dtype)[-32:,:,:-1,:].reshape(32, -1, 5)
            outputs = torch.tensor(dataset['controls'], device=self.device, dtype=self.dtype)[-32:,:,:,:].reshape(32, -1, 2)
        else:
            inputs  = torch.tensor(dataset['state_time'], device=self.device, dtype=self.dtype)[:-32,:,:-1,:].reshape(544, -1, 5)
            outputs = torch.tensor(dataset['controls'], device=self.device, dtype=self.dtype)[:-32,:,:,:].reshape(544, -1, 2)
        
        print('training data set:', not self.test)
        print('input size: ', inputs.shape)
        print('output size: ', outputs.shape) 

        self.xs = inputs.to(self.device)
        self.ys = outputs.to(self.device)
        
    def sample(self) -> Tuple[  torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        
        with torch.no_grad():
            n_functions = self.n_functions
            n_examples  = self.n_examples
            n_queries   = self.n_queries
            total_functions = self.xs.shape[0]
            
            function_indices = torch.randint(0, total_functions, (n_functions,), device=self.device)


            total_points = self.xs.shape[1]
            example_indices = torch.randint(0, total_points, (n_examples,), device=self.device)
            query_indices = torch.randint(0, total_points, (n_queries,), device=self.device)
            
            # Get function subset
            xs_subset = self.xs[function_indices]              # [n_functions, 5000, 5]
            ys_subset = self.ys[function_indices]              # [n_functions, 5000, 2]
            
            # Get point subset
            example_xs = xs_subset[:, example_indices, :]        # [n_functions, n_points, 5]
            example_ys = ys_subset[:, example_indices, :]        # [n_functions, n_points, 2]
            query_xs   = xs_subset[:, query_indices, :]
            query_ys   = ys_subset[:, query_indices, :]

            return example_xs, example_ys, query_xs, query_ys, {"function_indices":function_indices, "example_indices": example_indices, "query_indices": query_indices}
            