
#  Steps for Distillation
Note: In order to do the distillation in this process, we still need to be able to access to each environment. 

## 1 - Train Baselines


### 1.1) Generate datasets for SRL (random policy)

```
cd robotics-rl-srl
# Task_1: random target reaching 
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_random_simple --env OmnirobotEnv-v0 --simple-continual --num-episode 250 -f
# Task_2: Circular moving
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_circular --env OmnirobotEnv-v0 --circular-continual --num-episode 250 -f
```

### 1.2) Train SRL

```
cd srl_zoo
# Task_1: random target reaching 
python train.py --data-folder data/Omnibot_random_simple  -bs 32 --epochs 20 --state-dim 200 --training-set-size 20000 --losses autoencoder inverse
# Task_2: Circular moving
python train.py --data-folder data/Omnibot_circular  -bs 32 --epochs 20 --state-dim 200 --training-set-size 20000 --losses autoencoder inverse
```

### 1.3) Train policy

Train

```
cd ..
# Task_1: random target reaching 
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --srl-model-path srl_zoo/*path2srl_model* --num-timesteps 5000000 --env OmnirobotEnv-v0 --log-dir logs/simple/  --num-cpu 6 --simple-continual 

# Task_2: Circular moving
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --srl-model-path srl_zoo/*path2srl_model* --num-timesteps 5000000 --env OmnirobotEnv-v0 --log-dir logs/circular/  --num-cpu 6 --circular-continual

```

Visualize and plot (Optional)

```
# Visualize episodes 

python -m replay.enjoy_baselines --log-dir *file* --num-timesteps 10000 --render --action-proba
example : python -m replay.enjoy_baselines --log-dir logs/simple/OmnirobotEnv-v0/srl_combination/ppo2/19-04-25_10h19_42/ --num-timesteps 10000 --render --action-proba

# plot results
python -m replay.plots --log-dir /logs/simple/OmnirobotEnv-v0/srl_combination/ppo/ 

python -m replay.plots --log-dir /logs/circular/OmnirobotEnv-v0/srl_combination/ppo/ 

```

## 2 - Train Distillation


### 2.1) Generate dataset on Policy

In this step, we can either generate the dataset of (Observation, on_policy_proba_actions) or (reconstructed_observation, on_policy_proba_actions). 
The reconstructed_observations are given by decoding the observation given by the environment in to the srl_model that were used to train the RL.
That srl_model should be a type of generative-model like :vae, cvae or gan. So the ............................. 
In either case, each environment need to be accessible.



python -m environments.dataset_generator --name gen_test --env OmnirobotEnv-v0 --num-episode 10 --run-policy custom --log-custom-policy logs/srl_comb_generative_replayOmnirobotEnv-v0/srl_combination/ppo2/19-05-23_19h58_05/ --short-episodes --replay-generative-model vae -sc --log-generative-model srl_zoo/logs/Omnibot_random_simple//19-05-23_18h35_08_custom_cnn_ST_DIM200_vae_inverse/srl_model.pth


```
# Task_1: random target reaching 
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy logs/*path2policy* --short-episodes --save-path data/ --name reaching_on_policy -sc

# Task_2: Circular moving
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy logs/*path2policy* --short-episodes --save-path data/ --name circular_on_policy -cc

# Merge Datasets

(/ ! \ it removes the generated dataset for dataset 1 and 2)

python -m environments.dataset_merger --merge data/circular_on_policy/ data/reaching_on_policy/ data/merge_CC_SC

# Copy the merged Dataset to srl_zoo repository
cp -r data/merge_CC_SC srl_zoo/data/merge_CC_SC 
```

### 2.2) Train SRL 1&2 

```
cd srl_zoo
# Dataset 1
python train.py --data-folder data/merge_CC_SC  -bs 32 --epochs 20 --state-dim 200 --training-set-size 30000--losses autoencoder inverse

# Update your RL logs to load the proper SRL model for future distillation, i.e distillation: new-log/srl_model.pth
```

### 2.3) Run Distillation

```
# make a new log folder
mkdir logs/CL_SC_CC
cp config/srl_models_merged.yaml config/srl_models.yaml

# Merged Dataset 
python -m rl_baselines.train --algo distillation --srl-model raw_pixels --env OmnirobotEnv-v0 --log-dir logs/CL_SC_CC --teacher-data-folder srl_zoo/data/merge_CC_SC -cc --distillation-training-set-size 40000 --epochs-distillation 20 --latest
```

## 3 - Evaluation 

```
# Evaluation on task_1: random target reaching
python -m replay.enjoy_baselines --log-dir logs/*path2ditilled_policy* --num-timesteps 10000 --render --action-proba -sc
# Evaluation on task_2: Circular moving
python -m replay.enjoy_baselines --log-dir logs/*path2ditilled_policy* --num-timesteps 10000 --render --action-proba -cc

```
