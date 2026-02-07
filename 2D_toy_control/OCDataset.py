from typing import Tuple, Union
import os
import urllib.request
import tqdm
from scipy.io import loadmat

import torch
import numpy as np

from FunctionEncoder.Dataset.BaseDataset import BaseDataset

'''
Dataset for 2D center hill OC problem
16 different target state
200 trajectories randomly solved for each terminal state problem
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

if not os.path.exists("./Dataset/oc_velocity_dataset_structured.mat"):
    ln = "https://drive.usercontent.google.com/download?id=1Ccv99JpNld7HjR0jGVl53LA07kfroZam&export=download&authuser=0&confirm=t&uuid=965913be-a784-484a-b22d-2501b0da8433&at=ALoNOgmmgbY2vE8OZWy10vfDHYgf%3A1748885668556"
    desc = "2D CenterHill OC Problem"
    urllib.request.urlretrieve(ln, "./Dataset/oc_velocity_dataset_structured.mat", reporthook=show_progress)
print("Data set located.\n\n")

assert os.path.exists('./Dataset/oc_velocity_dataset_structured.mat'), "oc_velocity_dataset_structured.mat not found"
dataset = loadmat('./Dataset/oc_velocity_dataset_structured.mat')
inputs = dataset['positions_times']
outputs = dataset['velocities']
target_states = dataset['target_states']

print(f"There are {inputs.shape[0]} examples in the dataset, each corresponding to a different target state")
print(f"For each target state we generate {inputs.shape[1]} different trajectories over {inputs.shape[2]} time steps between 0 and 1 ")
print(f"Problem has dimension {inputs.shape[3]}")
print(f"target state in order, 16 in total {target_states}")


class OCDataset(BaseDataset):
    def __init__(self,
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
            n_functions = 10
        if n_examples is None and n_examples_per_sample is None:
            n_examples  = 500
        if n_queries is None and n_points_per_sample is None:
            n_queries   = 1000         
                 
        super().__init__(input_size=(3,),
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
                 
        inputs  = torch.tensor(dataset['positions_times'], device=self.device, dtype=self.dtype).reshape(16, -1, 3)
        print('input size: ', inputs.shape)
        outputs = torch.tensor(dataset['velocities'], device=self.device, dtype=self.dtype).reshape(16, -1, 2)
        print('output size: ', outputs.shape)        
                 
        self.xs = inputs.to(self.device)
        self.ys = outputs.to(self.device) 
        self.target_states = torch.tensor(dataset['target_states'], device=self.device, dtype=self.dtype)
            
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
            xs_subset = self.xs[function_indices]              # [n_functions, 4000, 3]
            ys_subset = self.ys[function_indices]              # [n_functions, 4000, 2]

            # Get point subset
            example_xs = xs_subset[:, example_indices, :]        # [n_functions, n_points, 3]
            example_ys = ys_subset[:, example_indices, :]        # [n_functions, n_points, 2]
            query_xs   = xs_subset[:, query_indices, :]
            query_ys   = ys_subset[:, query_indices, :]
            
            return example_xs, example_ys, query_xs, query_ys, {"function_indices":function_indices, "example_indices": example_indices, "query_indices": query_indices}