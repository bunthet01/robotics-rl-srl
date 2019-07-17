import argparse
import numpy as np
import torch as th
import os
import time
import cv2
import json

from datetime import datetime
from shutil import copyfile

from replay.enjoy_baselines import loadConfigAndSetup
from state_representation.models import loadSRLModel
from srl_zoo.preprocessing.utils import convertScalerToTensorAction, deNormalize
from srl_zoo.preprocessing.data_loader import DataLoaderCVAE

MAX_BATCH_SIZE_GPU = 64     # number of batch size before the gpu run out of memory
N_WORKERS = 1

def main():
    parser = argparse.ArgumentParser(description='Deteministic dataset generator for SRL training ' +
                                                 '(can be used for environment testing)')
    parser.add_argument('--save-path', type=str, default='data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--name', type=str, default='generated_reaching_on_policy', help='Folder name for the output')
    parser.add_argument('--no-record-data', action='store_true', default=False)
    # parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('--log-custom-policy', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--log-generative-model', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--short-episodes', action='store_true', default=False,
                        help='Generate short episodes (only 10 contacts with the target allowed).')
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--num-generated-sample', '--ngs',type=int, default='15625',help='The number of generated observation for the 4 class')
    args = parser.parse_args()

    # assert
    assert not (args.log_generative_model == '' and args.replay_generative_model == 'custom'), \
        "If using a custom policy, please specify a valid log folder for loading it."

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # File exists, need to deal with it
    if not args.no_record_data and os.path.exists(args.save_path + args.name):
        assert True, "Error: save directory '{}' already exists".format(args.save_path + args.name)

    os.mkdir(args.save_path + args.name)


    # load generative model
    generative_model = loadSRLModel(args.log_generative_model, th.cuda.is_available())
    generative_model_state_dim = generative_model.state_dim
    generative_model = generative_model.model.model
    
    # load configurations of rl model
    args.log_dir = args.log_custom_policy
    args.render = False
    args.shape_reward = False
    args.simple_continual, args.circular_continual, args.square_continual = False, False, False
    args.num_cpu = 1
    
    train_args, load_path, algo_name, algo_class, srl_model_path, env_kwargs_extra = loadConfigAndSetup(args)
    
    # load rl model
    model = algo_class.load(load_path)
    
    # check if the rl model was trained with SRL
    if srl_model_path!=None:
        srl_model = loadSRLModel(srl_model_path, th.cuda.is_available()).model.model
        

    # generate equal numbers of each action (decrete actions for 4 movement)         
    actions_0 = 0*np.ones(args.num_generated_sample)
    actions_1 = 1*np.ones(args.num_generated_sample)
    actions_2 = 2*np.ones(args.num_generated_sample)
    actions_3 = 3*np.ones(args.num_generated_sample)
    actions_unnormalize = np.concatenate((actions_0, actions_1, actions_2, actions_3)).astype(int)
    np.random.shuffle(actions_unnormalize)
    
    
    # create minibatchlist
    minibatchlist = DataLoaderCVAE.createTestMinibatchList(args.num_generated_sample*4, MAX_BATCH_SIZE_GPU)
    
    # data_loader
    data_loader = DataLoaderCVAE(minibatchlist, actions_unnormalize, generative_model_state_dim, n_workers=N_WORKERS, max_queue_len=1 )

    # some array for saving at the end  
    imgs_paths_array = []
    actions_array = []
    episode_starts = []

    #number of correct class prediction
    num_correct_class = 0

    for minibatch_num, (z, c) in enumerate(data_loader):
        if th.cuda.is_available():
            state = z.cuda()
            action = convertScalerToTensorAction(c).cuda()
        generated_obs = generative_model.decode(state, action)

        # save generated obervation 
        folder_path = os.path.join(args.save_path + args.name+"record_{:03d}/".format(minibatch_num))
        os.mkdir(folder_path)

        # Append the list of image's path and it's coressponding class (action)
        for i in range(generated_obs.size(0)):
            obs = deNormalize(generated_obs[i].to(th.device('cpu')).detach().numpy())
            obs = 255*obs[..., ::-1]
            imgs_paths = folder_path+"frame_{:06d}_class_{}.jpg".format(i, int(c[i]))
            cv2.imwrite(imgs_paths, obs.astype(int))  
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
        else:
            actions_proba_array = np.append(actions_proba_array,actions_proba, axis=0)            
        # count the correct predection 
        # used to evaluate the accuracy of the generative model
        for i in c-th.from_numpy(on_policy_actions).float():
            if i==0:
                num_correct_class +=1
            
    print("The generative model is {}% accurate for {} testing samples.".format(num_correct_class*100/(args.num_generated_sample*4), args.num_generated_sample*4))
    
    # save some data
    np.savez(args.save_path + args.name + "/preprocessed_data.npz", actions=actions_unnormalize.tolist(), actions_proba=actions_proba_array.tolist(), episode_starts=episode_starts, rewards = [])
    np.savez(args.save_path + args.name + "/ground_truth.npz", images_path=imgs_paths_array, ground_truth_states=[[]], target_positions=[[]])

    #save configs files
    copyfile(args.log_dir + "/env_globals.json", args.save_path + args.name+'/env_globals.json')
    with open(args.save_path + args.name+'/dataset_config.json', 'w') as f:
        json.dump({"img_shape" : train_args.get("img_shape", None)},f)

if __name__ == '__main__':
    main()
