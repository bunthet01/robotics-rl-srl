import argparse
import os

import numpy as np
import torch as th

from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import CnnPolicy

from torch.autograd import Variable

from environments.registry import registered_env
from real_robots.constants import *
from replay.enjoy_baselines import loadConfigAndSetup
from srl_zoo.preprocessing.utils import deNormalize, convertScalerToTensorAction
from state_representation.models import loadSRLModel, getSRLDim
from srl_zoo.utils import detachToNumpy
import matplotlib.pyplot as plt

VALID_POLICIES = [ 'random', 'ppo2', 'custom']
VALID_ACTIONS = [0, 1, 2, 3]

def main():
    parser = argparse.ArgumentParser(description='Image generator for CVAE ')
    parser.add_argument('--num-images', type=int, default=50, help='number of images to generate')
    parser.add_argument('--save-path', type=str, default='srl_zoo/data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--name', type=str, default='Generated_on_policy', help='Folder name for the output')
    parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('--log-generative-model', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--class-action', type=int, choices=VALID_ACTIONS, help='Class of actions'
                                                                                ' to generate the images')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Force the save, even if it overrides something else,' +
                             ' including partial parts if they exist')

    args = parser.parse_args()

    # assert (args.num_cpu > 0), "Error: number of cpu must be positive and non zero"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # File exists, need to deal with it
    if os.path.exists(args.save_path + args.name):
        assert args.force, "Error: save directory '{}' already exists".format(args.save_path + args.name)

        shutil.rmtree(args.save_path + args.name)
        for part in glob.glob(args.save_path + args.name + "_part-[0-9]*"):
            shutil.rmtree(part)

    os.mkdir(args.save_path + args.name)

    srl_model = loadSRLModel(args.log_generative_model, th.cuda.is_available(), env_object=None)
    srl_state_dim = srl_model.state_dim
    srl_model = srl_model.model.model

    imgs_paths_array = []
    actions_array = []

    np.random.seed(args.seed)
    print("Using seed:", args.seed)

    for i in np.arange(args.num_images):
        z = th.from_numpy(np.random.normal(0,1,srl_state_dim))[None,...].float()
        c = convertScalerToTensorAction([args.class_action])

        if th.cuda.is_available():
            z = z.cuda()
            c = c.cuda()
        generated_obs = srl_model.decode(z, c)
        generated_obs = deNormalize(detachToNumpy(generated_obs[0]))

        # save the images
        plt.imshow(generated_obs)
        imgs_paths = args.save_path + args.name+"/class_{}_frame{}.jpg".format(args.class_action, i)
        plt.savefig(imgs_paths)

        # Append the list of image's path and it's coressponding action
        imgs_paths_array.append(imgs_paths)
        actions_array.append(args.class_action)
    preprocessed_data = [imgs_paths_array, actions_array]
    
    # np.savez(args.save_path + args.name + "/ground_truth.npz", **ground_truth)
    np.savez(args.save_path + args.name + "/preprocessed_data.npz", preprocessed_data)


if __name__ == '__main__':
    main()
