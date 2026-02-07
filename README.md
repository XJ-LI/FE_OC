# FunctionEncoder for OC

This repository contains the relevant code  to the paper: [Link](https://arxiv.org/pdf/2509.18404)





## Commands for the experiments 

To train a model for the 2D trajectory planning example, run

```
cd 2D_toy_control
python train_ctrhill.py
python train_ctrhill_operator.py  --load_path ./		## optional step, not required
```

To evaluate pretrained models on new tasks, run the following scripts

```
python plotter_dataset_test.py
python plotter_dataset_test_operator.py				    ## optional step, not required
```


To train a model for the quadcopter example, run

```
cd Quadcopter_control
python train_ctrhill.py
python train_ctrhill_operator.py  --load_path ./		## optional step, not required
```

To evaluate pretrained models on new tasks, run the following scripts

```
python eval_quadcopter_test.py
python eval_quadcopter_test_operator.py				    ## optional step, not required
```



To train models for the bike  examples, run

```
cd Bike_control
python train_bike.py  --problem bike1  --epochs 70000  --hidden_size 256  --n_layers 6  --n_basis 150        ## for single obstacle case
python train_bike.py  --problem bike2  --epochs 70000  --hidden_size 256  --n_layers 6  --n_basis 200        ## for double obstacle case
```

To evaluate pretrained models on new tasks, run the following scripts

```
python eval_plotter_1_obs.py
python eval_plotter_2_obs.py
```

## Dependencies (What I Used)
* **FunctionEncoder** (==0.1.1)
* **PyTorch** (==2.8.0)
* **NumPy** (==1.26.4)
* **Scipy** (==1.16.3)
* **tqdm** (==4.67.2)
* **matplotlib** (==3.10.0)


