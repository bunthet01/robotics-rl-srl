


#  Steps for Distillation





# 1 - Train Baselines


### 0 - Generate datasets for SRL (random policy)

```
cd robotics-rl-srl
# Dataset 1 (random reaching target)
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_random_simple --env OmnirobotEnv-v0 --simple-continual --num-episode 250 -f
# Dataset 2 (Circular task)
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_circular --env OmnirobotEnv-v0 --circular-continual --num-episode 250 -f
```

### 1.1) Train SRL

```
cd srl_zoo
# Dataset 1 (random reaching target) 
python train.py --data-folder data/Omnibot_random_simple  -bs 32 --epochs 20 --state-dim 200 --losses arg.losses --figdir args.figdir --gpu-num 1
# Dataset 2 (Circular task)
python train.py --data-folder data/Omnibot_circular  -bs 32 --epochs 20 --state-dim 200  --losses arg.losses --figdir args.figdir --gpu-num 1

```
Choose a valid loose for arg.losses 

### 1.2) Train policy

Train

```
cd ..

# Dataset 1 (random reaching target)
python -m rl_baselines.train --algo ppo2 --srl-model arg.srl_model --num-timesteps 5000000 --env OmnirobotEnv-v0 --log-dir logs/simple/  --num-cpu 6 --simple-continual 

# Dataset 2 (Circular task)
python -m rl_baselines.train --algo ppo2 --srl-model arg.srl_model --num-timesteps 5000000 --env OmnirobotEnv-v0 --log-dir logs/circular/  --num-cpu 6 --circular-continual

```

Visualize and plot

```
# Visualize episodes 

python -m replay.enjoy_baselines --log-dir *file* --num-timesteps 10000 --render --action-proba
example : python -m replay.enjoy_baselines --log-dir logs/simple/OmnirobotEnv-v0/ground_truth/ppo2/19-04-25_10h19_42/ --num-timesteps 10000 --render --action-proba


# plot results
python -m replay.plots --log-dir /logs/simple/OmnirobotEnv-v0/ground_truth/ppo/ --latest

python -m replay.plots --log-dir /logs/circular/OmnirobotEnv-v0/groud_truth/ppo/ --latest

```

# 2 - Train Distillation


### 2.1) Generate dataset on Policy


```
# Dataset 1 (random reaching target)
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy logs/*path2policy* --save-path srl_zoo/data/ --name reaching_on_policy -sc --num-cpu 6

# Dataset 2 (Circular task)
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy logs/*path2policy* --save-path srl_zoo/data/ --name circular_on_policy -cc --num-cpu 6

# Merge Datasets

(/ ! \ it removes the generated dataset for dataset 1 and 2)

python -m environments.dataset_merger --merge srl_zoo/data/circular_on_policy/ srl_zoo/data/reaching_on_policy/ data/merge_CC_SC

```

### 2.2) Train SRL 1&2 

```
cd srl_zoo
# Dataset 1
python train.py --data-folder data/merge_CC_SC  -bs 32 --epochs 20 --state-dim 200 --training-set-size 30000--losses autoencoder inverse

# Update your RL logs to load the proper SRL model for future distillation, i.e distillation: new-log/srl_model.pth
```
```
BUNTHET: WE DONT DO THE 2.2 STEP

```

### 2.3) Run Distillation

```
# Merged Dataset 
python -m rl_baselines.train --algo distillation --srl-model raw_pixels --env OmnirobotEnv-v0 --log-dir logs/CL_SC_CC --teacher-data-folder merge_CC_SC -cc --epochs-distillation 20
```
