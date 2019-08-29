#"""
#This is the script to generate dataset for policy distillation without having to access to each teacher's environment.
#What we need is only the generative model(cvae, cvae_new, cgan,cgan_new,gan, vae) and policy model of each teacher task (and the srl model that used to train each teacher's RL task.)
# It is designed for now specefically for Omnirobot_env
#"""

import argparse
import numpy as np
import torch as th
import os
import time
import cv2
import json

from datetime import datetime
from shutil import copyfile
from tqdm import tqdm

from replay.enjoy_baselines import loadConfigAndSetup
from state_representation.models import loadSRLModel
from srl_zoo.preprocessing.utils import one_hot, deNormalize
from srl_zoo.preprocessing.data_loader import DataLoaderConditional
from real_robots.constants import TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y, MAX_X, MIN_X, MAX_Y, MIN_Y   # using Omnibot_env 

MAX_BATCH_SIZE_GPU = 128 # number of batch size before the gpu run out of memory	

def main():
    parser = argparse.ArgumentParser(description='Deteministic dataset generator for SRL training ' +
                                                 '(can be used for environment testing)')
    parser.add_argument('--save-path', type=str, default='data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--name', type=str, default='generated_reaching_on_policy', help='Folder name for the output')
    parser.add_argument('--no-record-data', action='store_true', default=False)
    parser.add_argument('--log-custom-policy', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--log-generative-model', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--short-episodes', action='store_true', default=False,
                        help='Generate short episodes (only 2 contacts with the target allowed).')	# we can change the number of contact in omnirobot_env.py
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--ngsa','--num-generating-samples-per-action', type=int, default='2000',
                        help='The number of generated observation for each of the 4 class')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('--task', type=str, default=None, choices=['sc','cc'], help='choose task for data set generation')
    parser.add_argument('--grid-walker', action='store_true', default=False,
                        help='Generate the robot as grid walker.')
    parser.add_argument('--gw-step', type=int, default=0.1, help='the grid walker step') 
    parser.add_argument('--gw-episode', type=int, default=100, help='number of episode in the grid walker ')                
                        

	

    args = parser.parse_args()

    # assert
    assert not (args.log_generative_model == '' and args.replay_generative_model == 'custom'), \
        "If using a custom policy, please specify a valid log folder for loading it."
    assert not (args.task is None), "must choose a task"


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # File exists, need to deal with it
    assert not os.path.exists(args.save_path + args.name),"Error: save directory '{}' already exists".format(args.save_path + args.name)
    os.mkdir(args.save_path + args.name)
    
    print("Using seed = ", args.seed)
    np.random.seed(args.seed)

    # load generative model
    generative_model, gernerative_model_losses, only_action = loadSRLModel(args.log_generative_model, th.cuda.is_available())
    generative_model_state_dim = generative_model.state_dim
    generative_model = generative_model.model.model
    
    # load configurations of rl model
    args.log_dir = args.log_custom_policy
    args.render = args.display
    args.simple_continual, args.circular_continual, args.square_continual = False, False, False
    args.num_cpu = 1
    
    train_args, load_path, algo_name, algo_class, srl_model_path, env_kwargs_extra = loadConfigAndSetup(args)
    
    # assert that the RL was not trained with ground_truth, since the generative model dont 
    # generate ground_truth to forward to RL policy model that take input as ground truth
    assert not ( train_args['srl_model'] == "ground_truth"), "can not use RL model trained with ground_truth"
    
    # load rl model
    model = algo_class.load(load_path)
    
    # check if the rl model was trained with SRL
    if srl_model_path!=None:
        srl_model,_,_ = loadSRLModel(srl_model_path, th.cuda.is_available())
        srl_model = srl_model.model.model
    # some condtion    
    using_conditional_model = "cgan" in gernerative_model_losses or "cvae" in gernerative_model_losses or "cgan_new" in gernerative_model_losses
    use_cvae_new = "cvae_new" in gernerative_model_losses
        
    # generate equal numbers of each action (decrete actions for 4 movement) 
    if not args.grid_walker and not use_cvae_new:         
        actions_0 = 0*np.ones(args.ngsa)
        actions_1 = 1*np.ones(args.ngsa)
        actions_2 = 2*np.ones(args.ngsa)
        actions_3 = 3*np.ones(args.ngsa)
        actions = np.concatenate((actions_0, actions_1, actions_2, actions_3)).astype(int)
        np.random.seed(args.seed)
        np.random.shuffle(actions)
    else: 
        actions = th.zeros(1)

    # create minibatchlist and grid list and target's position list
    grid_list = []
    target_pos_list = []
    if not args.grid_walker:
        minibatchlist = DataLoaderConditional.createTestMinibatchList(args.ngsa*4, MAX_BATCH_SIZE_GPU)
    else:
        grid_walker_number = (int((MAX_X-MIN_X)/args.gw_step)-1)*(int((MAX_Y-MIN_Y)/args.gw_step)-1)
        minibatchlist = DataLoaderConditional.createTestMinibatchList(grid_walker_number*args.gw_episode, MAX_BATCH_SIZE_GPU)   
        # create grid list for grid walker
        for _ in range(args.gw_episode):
            for i in range(int((MAX_X-MIN_X)/args.gw_step)-1):
                for j in range(int((MAX_Y-MIN_Y)/args.gw_step)-1):
                    x = MAX_X-(i+1)*args.gw_step
                    y = MAX_Y-(j+1)*args.gw_step
                    grid_list.append([x,y])
        # create target list for each episode 
        for _ in range(args.gw_episode):
            random_init_x = np.random.random_sample(1).item() * (TARGET_MAX_X - TARGET_MIN_X) + \
                        TARGET_MIN_X if args.task == 'sc' else 0
            random_init_y = np.random.random_sample(1).item() * (TARGET_MAX_Y - TARGET_MIN_Y) + \
                        TARGET_MIN_Y if args.task == 'sc' else 0
            for _ in range(int((MAX_X-MIN_X)/args.gw_step)-1):
                for _ in range(int((MAX_Y-MIN_Y)/args.gw_step)-1):
                    target_pos_list.append([random_init_x, random_init_y])
    grid_list = np.asarray(grid_list)
    target_pos_list =np.asarray(target_pos_list)
            
        
    # data_loader
    data_loader = DataLoaderConditional(minibatchlist, actions,args.task, generative_model_state_dim,TARGET_MAX_X, TARGET_MIN_X, TARGET_MAX_Y, TARGET_MIN_Y, MAX_X, MIN_X, MAX_Y, MIN_Y,args.grid_walker, grid_list,target_pos_list, seed = args.seed, max_queue_len=4 )

    # some lists for saving at the end  
    imgs_paths_array = []
    actions_array = []
    episode_starts = []
    
    

    #number of correct class prediction
    num_correct_class = np.zeros(4)
    pbar = tqdm(total=len(minibatchlist))
    for minibatch_num, (z, c, t, r) in enumerate(data_loader):       
        if th.cuda.is_available():
            state = z.to('cuda')
            action = c.to('cuda')
            target = t.to('cuda') 
            robot_pos = r.to('cuda')
        if using_conditional_model:
            generated_obs = generative_model.decode(state, action, target, only_action)
        elif use_cvae_new:
            generated_obs = generative_model.decode(state, target,robot_pos)
        else:
            generated_obs = generative_model.decode(state)            

        # save generated obervation 
        # [TODO]: even thought we name it "record" but it does not yet seperates images between episodes, we just save every minibatch_num
        folder_path = os.path.join(args.save_path + args.name+"record_{:03d}/".format(minibatch_num))
        os.mkdir(folder_path)

        # Append the list of image's path and it's coressponding class (action)
        for i in range(generated_obs.size(0)):
            
            obs = deNormalize(generated_obs[i].to(th.device('cpu')).detach().numpy())
            obs = 255*obs[..., ::-1]
            if using_conditional_model:
                if only_action:
                    imgs_paths = folder_path+"frame_{:06d}_class_{}.jpg".format(i, int(c[i]))
                else:
                    imgs_paths = folder_path+"frame_{:06d}_class_{}_tp_{:.2f}_{:.2f}.jpg".format(i, int(c[i]),t[i][0], t[i][1]) 
            elif use_cvae_new:
                imgs_paths = folder_path+"frame_{:06d}_tp_{:.2f}_{:.2f}_rp_{:.2f}_{:.2f}.jpg".format(i,t[i][0], t[i][1],r[i][0], r[i][1])             
            else:
                imgs_paths = folder_path+"frame_{:06d}.jpg".format(i)
                
            cv2.imwrite(imgs_paths, obs.astype(np.uint8))  
            imgs_paths_array.append(imgs_paths)
            if i==0 and minibatch_num==0 :
                episode_starts.append(True)
            else:
                episode_starts.append(False)
                
        if srl_model_path!=None:
            generated_obs_state = srl_model.forward(generated_obs.cuda()).to(th.device('cpu')).detach().numpy()
            on_policy_actions = model.getAction(generated_obs_state)
            actions_proba = model.getActionProba(generated_obs_state)
        else:
            on_policy_actions = model.getAction(generated_obs.to(th.device('cpu')).detach().numpy())
            actions_proba = model.getActionProba(generated_obs.to(th.device('cpu')).detach().numpy()) 
            
            
        
        if minibatch_num == 0:
            actions_proba_array = actions_proba
            on_policy_actions_array = on_policy_actions
            z_array = z.detach().numpy()
            
        else:
            actions_proba_array = np.append(actions_proba_array,actions_proba, axis=0)  
            on_policy_actions_array = np.append(on_policy_actions_array,on_policy_actions, axis=0 )
            z_array = np.append(z_array,z.detach().numpy(), axis=0)   
                   
        # count the correct predection 
        # used to evaluate the accuracy of the generative model
        
        if using_conditional_model:
            for i in np.arange(on_policy_actions.shape[0]):
                if c.numpy()[i]==on_policy_actions[i]:
                    if c[i] == 0:num_correct_class[0] += 1
                    elif c[i] == 1:num_correct_class[1] += 1
                    elif c[i] == 2:num_correct_class[2] += 1
                    else: num_correct_class[3] += 1
        pbar.update(1)
    pbar.close()
                
    if using_conditional_model:         
        correct_observations = (100/args.ngsa)*num_correct_class        
        print("The generative model is {}% accurate for {} testing samples.".format(np.sum(num_correct_class)*100/(args.ngsa*4), args.ngsa*4))
        print("Correct observations of action class '0' : {}%".format(correct_observations[0]))
        print("Correct observations of action class '1' : {}%".format(correct_observations[1]))
        print("Correct observations of action class '2' : {}%".format(correct_observations[2]))
        print("Correct observations of action class '3' : {}%".format(correct_observations[3]))
    
    # save some data
    	# We dont have any information about the ground_truth_states and target_positions. 
    	# They are saved for the sake of not causing error in data merging only,since data merging looks also to merge these two arrays.  

    np.savez(args.save_path + args.name + "/preprocessed_data.npz", actions=on_policy_actions_array.tolist(), actions_proba=actions_proba_array.tolist(), episode_starts=episode_starts, rewards = [], z_array=z_array.tolist())
    np.savez(args.save_path + args.name + "/ground_truth.npz", images_path=imgs_paths_array, ground_truth_states=[[]], target_positions=[[]])   

    #save configs files
    copyfile(args.log_dir + "/env_globals.json", args.save_path + args.name+'/env_globals.json')
    with open(args.save_path + args.name+'/dataset_config.json', 'w') as f:
        json.dump({"img_shape" : train_args.get("img_shape", None)},f)
    if using_conditional_model:  
        with open(args.save_path + args.name+'/class_eval.json', 'w') as f:
            json.dump({"num_correct_class" : correct_observations.tolist(), "ngsa_per_class":args.ngsa, "random_seed":args.seed},f)
    else: 
        with open(args.save_path + args.name+'/class_eval.json', 'w') as f:
            json.dump({"random_seed":args.seed},f)

if __name__ == '__main__':
    main()
