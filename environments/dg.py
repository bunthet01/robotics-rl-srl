from __future__ import division, absolute_import, print_function

import argparse
import glob
import json
import multiprocessing
import os
import shutil
import time

import numpy as np
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import CnnPolicy
import tensorflow as tf
import torch as th
from torch.autograd import Variable

from environments import ThreadingType
from environments.registry import registered_env
from environments.utils import makeEnv
from real_robots.constants import *
from replay.enjoy_baselines import createEnv, loadConfigAndSetup
from rl_baselines.utils import MultiprocessSRLModel, loadRunningAverage
from srl_zoo.utils import printRed, printYellow
from srl_zoo.preprocessing.utils import deNormalize
from state_representation.models import loadSRLModel, getSRLDim
from PIL import Image
from srl_zoo.utils import detachToNumpy
from .utils import convertScalerToVectorAction
import matplotlib.pyplot as plt

RENDER_HEIGHT = 224
RENDER_WIDTH = 224
VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae","cvae", "dae", "random"]
VALID_POLICIES = [ 'random', 'ppo2', 'custom']
VALID_ACTIONS = [0, 1, 2, 3]

def main():
    parser = argparse.ArgumentParser(description='Image generator for CVAE ')
    parser.add_argument('--num-cpu', type=int, default=1, help='number of cpu to run on')
    parser.add_argument('--num-images', type=int, default=50, help='number of images to generate')
    parser.add_argument('--save-path', type=str, default='srl_zoo/data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--name', type=str, default='kuka_button', help='Folder name for the output')
    parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('--log-generative-model', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--class-action', type=int, choices=VALID_ACTIONS, help='Class of actions'
                                                                                ' to generate the images')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Force the save, even if it overrides something else,' +
                             ' including partial parts if they exist')

    args = parser.parse_args()

    assert (args.num_cpu > 0), "Error: number of cpu must be positive and non zero"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # this is done so seed 0 and 1 are different and not simply offset of the same datasets.
    args.seed = np.random.RandomState(args.seed).randint(int(1e10))
    print("args.seed :", args.seed)

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

    img_path_array = []
    actions_array = []

    # dataset_config_fp = open(args.save_path + args.name + "/dataset_config.json", 'w')
    # env_globals = open(args.save_path + args.name + "/env_globals.json", 'w')
    # t = True
    for i in range(args.num_images):
        z = th.normal(0,1,srl_state_dim).float()
        c = convertScalerToVectorAction(th.tensor([args.class_action]))

        if th.cuda.is_available():
            z = z.cuda()
            c = c.cuda()
        generated_obs = srl_model.decode_cvae(z, c)
        generated_obs = deNormalize(detachToNumpy(generated_obs[0]))

        # save the images
        plt.imshow(generated_obs)
        img_path = args.save_path + args.name+"/class_{}_frame{}.jpg".format(args.class_action, i)
        plt.savefig(img_path)

        # Append the list of image's path and it's coressponding action
        img_path_array.append(img_path)
        actions_array.append(args.class_action)
    preprocessed_data = [img_path_array, actions_array]
    
    # np.savez(args.save_path + args.name + "/ground_truth.npz", **ground_truth)
    np.savez(args.save_path + args.name + "/preprocessed_data.npz", preprocessed_data)


if __name__ == '__main__':
    main()
