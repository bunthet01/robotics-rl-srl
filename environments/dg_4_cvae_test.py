import argparse
import numpy as np
import torch as th
import os
from datetime import datetime
import environments.omnirobot_gym.omnirobot_env as omnirobot_env
import matplotlib.pyplot as plt
import time
import cv2

from environments.registry import registered_env
from replay.enjoy_baselines import loadConfigAndSetup
from state_representation.models import loadSRLModel
from srl_zoo.preprocessing.utils import convertScalerToTensorAction, deNormalize
from srl_zoo.preprocessing.data_loader import DataLoader

VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae", "dae", "random"]
VALID_POLICIES = ['walker', 'random', 'ppo2', 'custom']
VALID_ACTIONS = [0, 1, 2, 3]
MAX_BATCH_SIZE_GPU = 4



def main():
    parser = argparse.ArgumentParser(description='Deteministic dataset generator for SRL training ' +
                                                 '(can be used for environment testing)')
    parser.add_argument('--save-path', type=str, default='data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--name', type=str, default='generated_reaching_on_policy', help='Folder name for the output')
    parser.add_argument('--no-record-data', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0, help='the seed')
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

    # this is done so seed 0 and 1 are different and not simply offset of the same datasets.
    # args.seed = np.random.RandomState(args.seed).randint(int(1e10))
    # print("args.seed :", args.seed)

    # File exists, need to deal with it
    if not args.no_record_data and os.path.exists(args.save_path + args.name):
        assert True, "Error: save directory '{}' already exists".format(args.save_path + args.name)

    os.mkdir(args.save_path + args.name)
    os.mkdir(args.save_path + args.name+"generated_images/")


    # load generative model
    generative_model = loadSRLModel(args.log_generative_model, th.cuda.is_available())
    generative_model_state_dim = generative_model.state_dim
    generative_model = generative_model.model.model


    # generate observations        
    actions_0 = 0*np.ones(args.num_generated_sample)
    actions_1 = 1*np.ones(args.num_generated_sample)
    actions_2 = 2*np.ones(args.num_generated_sample)
    actions_3 = 3*np.ones(args.num_generated_sample)
    actions_unnormalize = np.concatenate((actions_0, actions_1, actions_2, actions_3)).astype(int)
    np.random.shuffle(actions_unnormalize)
    actions = convertScalerToTensorAction(actions_unnormalize)
    
    # if th.cuda.is_available():
    #     generated_obs = th.zeros((z.size(0),3,128,128)).cuda()
    # else:
    #     generated_obs = th.zeros((z.size(0),3,128,128))
    generated_obs = th.zeros((z.size(0),3,128,128)).cuda()

    print('Generating oberservation ...', end='')
    start_time = time.time()
    minibatchlist = DataLoader.createTestMinibatchList(args.num_generated_sample*4, MAX_BATCH_SIZE_GPU)
    for i in np.arange(len(minibatchlist)-1):
        z = th.from_numpy(np.random.normal(0,1,(minibatchlist[i].shape[0], generative_model_state_dim))).float()
        if th.cuda.is_available():
            generated_obs[minibatchlist[i]] = generative_model.decode(z.cuda(),actions[minibatchlist[i]].cuda())
            th.cuda.empty_cache()
        else:
            generated_obs[minibatchlist[i]] = generative_model.decode(z,actions[minibatchlist[i]])
    end_time = time.time()
    print("{0:.2f} seconds.".format(end_time-start_time))


    # save the images
    print('Saving generated datasets ...', end='')
    start_time = time.time()
    imgs_paths_array = []
    actions_array = []
    episode_starts = []
            
    for i in np.arange(args.num_generated_sample*4):
        if i%250 == 0:
            folder_path = os.path.join(args.save_path + args.name+"generated_images/image_folder_{:03d}/".format(int(i/250)))
            os.mkdir(folder_path)
        obs = deNormalize(generated_obs[i].to(th.device('cpu')).detach().numpy())
        obs = 255*obs[..., ::-1]
        imgs_paths = folder_path+"image_{:03d}_class_{}.jpg".format(int(i%250), actions_unnormalize[i])
        cv2.imwrite(imgs_paths, obs.astype(int))

        # Append the list of image's path and it's coressponding action
        imgs_paths_array.append(imgs_paths)
        if i==0:
            episode_starts.append(True)
        else:
            episode_starts.append(False)
    end_time = time.time()
    print("{0:.2f} seconds.".format(end_time-start_time))
    
    # load configurations of rl model
    print('Loading RL model ...')
    args.log_dir = args.log_custom_policy
    args.render = False
    args.shape_reward = False
    args.simple_continual, args.circular_continual, args.square_continual = False, False, False
    args.num_cpu = 1
    
    train_args, load_path, algo_name, algo_class, srl_model_path, env_kwargs_extra = loadConfigAndSetup(args)
    
    #load rl model
    model = algo_class.load(load_path)
    
    # check if the rl model was trained with SRL
    if srl_model_path!=None:
        srl_model = loadSRLModel(srl_model_path, th.cuda.is_available()).model.model
        generated_obs_state = srl_model.forward(generated_obs.cuda()).to(th.device('cpu')).detach().numpy()
        on_policy_actions = model.getAction(generated_obs_state)
        actions_proba = model.getActionProba(generated_obs_state) 
    else:
        generated_obs = generated_obs.to(th.device('cpu')).detach().numpy()
        on_policy_actions = model.getAction(generated_obs)
        actions_proba = model.getActionProba(generated_obs)
        
    # Check the accuracy of the model
    print('Checking generative model accuracy ...')
    true_action = 0
    for i in actions_unnormalize-on_policy_actions:
        if i==0:
            true_action +=1
            
    print("The generative model is {}% accurate for {} testing samples.".format(true_action*100/(args.num_generated_sample*4), args.num_generated_sample*4))

    np.savez(args.save_path + args.name + "/preprocessed_data.npz", actions=actions_unnormalize.tolist(), actions_proba=actions_proba, episode_starts=episode_starts)
    np.savez(args.save_path + args.name + "/ground_truth.npz", images_path=imgs_paths_array)
    th.cuda.empty_cache()
        
if __name__ == '__main__':
    main()
