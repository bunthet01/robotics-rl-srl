
"""
Train script for RL algorithms
"""
import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pprint import pprint

import yaml
from stable_baselines.common import set_global_seeds
from visdom import Visdom

from environments.registry import registered_env
from environments.srl_env import SRLGymEnv
from rl_baselines import AlgoType, ActionType
from rl_baselines.registry import registered_rl
from rl_baselines.utils import computeMeanReward
from rl_baselines.utils import filterJSONSerializableObjects
from rl_baselines.visualize import timestepsPlot, episodePlot,episodesEvalPlot
from rl_baselines.cross_eval import episodeEval
from srl_zoo.utils import printGreen, printYellow
from state_representation import SRLType
from state_representation.registry import registered_srl

VISDOM_PORT = 8097
LOG_INTERVAL = 0  # initialised during loading of the algorithm
LOG_DIR = ""
ALGO = None
ALGO_NAME = ""
ENV_NAME = ""
PLOT_TITLE = ""
EPISODE_WINDOW = 40  # For plotting moving average
EVAL_TASK=['cc','sc','sqc']
CROSS_EVAL = True
EPISODE_WINDOW_DISTILLATION_WIN = 20
NEW_LR=0.001


viz = None
n_steps = 0
SAVE_INTERVAL = 0  # initialised during loading of the algorithm
N_EPISODES_EVAL = 100  # Evaluate the performance on the last 100 episodes
MIN_EPISODES_BEFORE_SAVE = 100  # Number of episodes to train on before saving best model
params_saved = False
best_mean_reward = -10000

win, win_smooth, win_episodes, win_crossEval= None, None, None, None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


def saveEnvParams(kuka_env_globals, env_kwargs):
    """
    :param kuka_env_globals: (dict)
    :param env_kwargs: (dict) The extra arguments for the environment
    """
    params = filterJSONSerializableObjects({**kuka_env_globals, **env_kwargs})
    with open(LOG_DIR + "env_globals.json", "w") as f:
        json.dump(params, f)


def latestPath(path):
    """
    :param path: path to the log folder (defined in srl_model.yaml) (str)
    :return: path to latest learned model in the same dataset folder (str)
    """
    return max(
        [path + "/" + d for d in os.listdir(path) if not d.startswith('baselines') and os.path.isdir(path + "/" + d)],
        key=os.path.getmtime) + '/srl_model.pth'


def configureEnvAndLogFolder(args, env_kwargs, all_models):
    """
    :param args: (ArgumentParser object)
    :param env_kwargs: (dict) The extra arguments for the environment
    :param all_models: (dict) The location of all the trained SRL models
    :return: (ArgumentParser object, dict)
    """
    global PLOT_TITLE, LOG_DIR
    # Reward sparse or shaped
    env_kwargs["shape_reward"] = args.shape_reward
    # Actions in joint space or relative position space
    env_kwargs["action_joints"] = args.action_joints
    args.log_dir += args.env + "/"

    models = all_models[args.env] ## models: config file dict of srl_model path
    PLOT_TITLE = args.srl_model
    path = models.get(args.srl_model)
    args.log_dir += args.srl_model + "/"

    env_kwargs["srl_model"] = args.srl_model
    if registered_srl[args.srl_model][0] == SRLType.SRL:
        env_kwargs["use_srl"] = True
        if args.latest:
            printYellow("Using latest srl model in {}".format(models['log_folder']))
            env_kwargs["srl_model_path"] = latestPath(models['log_folder'])
        else:
            assert path is not None, "Error: SRL path not defined for {} in {}".format(args.srl_model,
                                                                                       args.srl_config_file)
            # Path depending on whether to load the latest model or not
            if args.srl_model_path is not None: ## [HACK way to pass srl model weights path to terminal]
                assert os.path.exists(args.srl_model_path), "SRL model weights: {} doesn't exist.".format(args.srl_model_path)
                env_kwargs["srl_model_path"] = args.srl_model_path
            else:
                srl_model_path = models['log_folder'] + path
                env_kwargs["srl_model_path"] = srl_model_path
    # Use of continual learning env
    env_kwargs["simple_continual_target"] = args.simple_continual
    env_kwargs["circular_continual_move"] = args.circular_continual
    env_kwargs["square_continual_move"] = args.square_continual
    env_kwargs["eight_continual_move"] = args.eight_continual

    # Add date + current time
    args.log_dir += "{}/{}/".format(ALGO_NAME, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    LOG_DIR = args.log_dir
    # wait one second if the folder exist to avoid overwritting logs
    time.sleep(1)
    os.makedirs(args.log_dir, exist_ok=True)

    return args, env_kwargs


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global win, win_smooth, win_episodes, win_crossEval, n_steps, viz, params_saved, best_mean_reward
    # Create vizdom object only if needed
    if viz is None:
        viz = Visdom(port=VISDOM_PORT)

    is_es = registered_rl[ALGO_NAME][1] == AlgoType.EVOLUTION_STRATEGIES

    # Save RL agent parameters
    if not params_saved:
        # Filter locals
        params = filterJSONSerializableObjects(_locals)
        with open(LOG_DIR + "rl_locals.json", "w") as f:
            json.dump(params, f)
        params_saved = True

    # Save the RL model if it has improved
    if (n_steps + 1) % SAVE_INTERVAL == 0:
        # Evaluate network performance
        ok, mean_reward = computeMeanReward(LOG_DIR, N_EPISODES_EVAL, is_es=is_es, return_n_episodes=True)
        if ok:
            # Unpack mean reward and number of episodes
            mean_reward, n_episodes = mean_reward
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
        else:
            # Not enough episode
            mean_reward = -10000
            n_episodes = 0

        # Save Best model
        if mean_reward > best_mean_reward and n_episodes >= MIN_EPISODES_BEFORE_SAVE:
            # Try saving the running average (only valid for mlp policy)
            try:
                if 'env' in _locals:
                    _locals['env'].save_running_average(LOG_DIR)
                else:
                    _locals['self'].env.save_running_average(LOG_DIR)
            except AttributeError:
                pass

            best_mean_reward = mean_reward
            printGreen("Saving new best model")
            ALGO.save(LOG_DIR + ALGO_NAME + "_model.pkl", _locals)

        if n_episodes >= 0:

            # For every checkpoint, we create one directory for saving logs file (policy and run mean std)
            if EPISODE_WINDOW_DISTILLATION_WIN > 0:
                if n_episodes % EPISODE_WINDOW_DISTILLATION_WIN == 0:
                    ALGO.save(LOG_DIR + ALGO_NAME + '_' + str(n_episodes) + "_model.pkl", _locals)
                    if CROSS_EVAL:  # If we want to do the cross evaluation after the training
                        eps_path = LOG_DIR + "model_" + str(n_episodes)
                        try:
                            os.mkdir(LOG_DIR + "model_" + str(n_episodes))
                        except OSError:
                            pass
                            #print("Creation of the directory {} failed".format(eps_path))

                        ALGO.save("{}/{}".format(eps_path, ALGO_NAME + "_model.pkl"), _locals)
                        try:
                            if 'env' in _locals:
                                _locals['env'].save_running_average(eps_path)
                            else:
                                _locals['self'].env.save_running_average(eps_path)
                        except AttributeError:
                            pass

    # Plots in visdom
    if viz and (n_steps + 1) % LOG_INTERVAL == 0:
        win = timestepsPlot(viz, win, LOG_DIR, ENV_NAME, ALGO_NAME, bin_size=1, smooth=0, title=PLOT_TITLE, is_es=is_es)
        win_smooth = timestepsPlot(viz, win_smooth, LOG_DIR, ENV_NAME, ALGO_NAME, title=PLOT_TITLE + " smoothed",
                                   is_es=is_es)
        win_episodes = episodePlot(viz, win_episodes, LOG_DIR, ENV_NAME, ALGO_NAME, window=EPISODE_WINDOW,
                                   title=PLOT_TITLE + " [Episodes]", is_es=is_es)
    n_steps += 1
    return True


def main():
    # Global variables for callback
    global ENV_NAME, ALGO, ALGO_NAME, LOG_INTERVAL, VISDOM_PORT, viz
    global SAVE_INTERVAL, EPISODE_WINDOW, MIN_EPISODES_BEFORE_SAVE
    parser = argparse.ArgumentParser(description="Train script for RL algorithms")
    parser.add_argument('--algo', default='ppo2', choices=list(registered_rl.keys()), help='RL algo to use',
                        type=str)
    parser.add_argument('--env', type=str, help='environment ID', default='KukaButtonGymEnv-v0',
                        choices=list(registered_env.keys()))
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--episode-window', type=int, default=40,
                        help='Episode window for moving average plot (default: 40)')
    parser.add_argument('--log-dir', default='/tmp/gym/', type=str,
                        help='directory to save agent logs and model (default: /tmp/gym)')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--srl-model', type=str, default='raw_pixels', choices=list(registered_srl.keys()),
                        help='SRL model to use')
    parser.add_argument('--num-stack', type=int, default=1, help='number of frames to stack (default: 1)')
    parser.add_argument('--action-repeat', type=int, default=1,
                        help='number of times an action will be repeated (default: 1)')
    parser.add_argument('--port', type=int, default=8097, help='visdom server port (default: 8097)')
    parser.add_argument('--no-vis', action='store_true', default=False, help='disables visdom visualization')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('-c', '--continuous-actions', action='store_true', default=False)
    parser.add_argument('-joints', '--action-joints', action='store_true', default=False,
                        help='set actions to the joints of the arm directly, instead of inverse kinematics')
    parser.add_argument('-r', '--random-target', action='store_true', default=False,
                        help='Set the button to a random position')
    parser.add_argument('--srl-config-file', type=str, default="config/srl_models.yaml",
                        help='Set the location of the SRL model path configuration.')
    parser.add_argument('--hyperparam', type=str, nargs='+', default=[])
    parser.add_argument('--min-episodes-save', type=int, default=100,
                        help="Min number of episodes before saving best model")
    parser.add_argument('--latest', action='store_true', default=False,
                        help='load the latest learned model (location:srl_zoo/logs/DatasetName/)')
    parser.add_argument('--load-rl-model-path', type=str, default=None,
                        help="load the trained RL model, should be with the same algorithm type")
    parser.add_argument('-sc', '--simple-continual', action='store_true', default=False,
                        help='Simple red square target for task 1 of continual learning scenario. ' +
                             'The task is: robot should reach the target.')
    parser.add_argument('-cc', '--circular-continual', action='store_true', default=False,
                        help='Blue square target for task 2 of continual learning scenario. ' +
                             'The task is: robot should turn in circle around the target.')
    parser.add_argument('-sqc', '--square-continual', action='store_true', default=False,
                        help='Green square target for task 3 of continual learning scenario. ' +
                             'The task is: robot should turn in square around the target.')
    parser.add_argument('-ec', '--eight-continual', action='store_true', default=False,
                        help='Green square target for task 4 of continual learning scenario. ' +
                             'The task is: robot should do the eigth with the target as center of the shape.')
    parser.add_argument('--teacher-data-folder', type=str, default="",
                        help='Dataset folder of the teacher(s) policy(ies)', required=False)
    parser.add_argument('--epochs-distillation', type=int, default=30, metavar='N',
                        help='number of epochs to train for distillation(default: 30)')
    parser.add_argument('--distillation-training-set-size', type=int, default=-1,
                        help='Limit size (number of samples) of the training set (default: -1)')
    parser.add_argument('--perform-cross-evaluation-cc', action='store_true', default=False,
                        help='A cross evaluation from the latest stored model to all tasks')
    parser.add_argument('--eval-episode-window', type=int, default=400, metavar='N',
                        help='Episode window for saving each policy checkpoint for future distillation(default: 100)')
    parser.add_argument('--new-lr', type=float, default=1.e-4,
                        help="New learning rate ratio to train a pretrained agent")
    parser.add_argument('--img-shape', type=str, default="(3,64,64)",
                        help="Image shape of environment.")
    parser.add_argument("--gpu-num", help="Choose the number of GPU (CUDA_VISIBLE_DEVICES).",
                        type=str, default="1", choices=["0", "1", "2", "3", "5", "6", "7", "8"])
    parser.add_argument("--srl-model-path", help="SRL model weights path",
                        type=str, default=None)
    parser.add_argument("--relative-pos", action='store_true', default=False,
                        help="For 'ground_truth': use relative position or not.")
    # Ignore unknown args for now
    args, unknown = parser.parse_known_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    env_kwargs = {}
    if args.img_shape is None:
        img_shape = None #(3,224,224)
    else:
        img_shape = tuple(map(int, args.img_shape[1:-1].split(",")))
    env_kwargs['img_shape'] = img_shape
    # LOAD SRL models list
    assert os.path.exists(args.srl_config_file), \
        "Error: cannot load \"--srl-config-file {}\", file not found!".format(args.srl_config_file)
    with open(args.srl_config_file, 'rb') as f:
        all_models = yaml.load(f)
    # Sanity check
    assert args.episode_window >= 1, "Error: --episode_window cannot be less than 1"
    assert args.num_timesteps >= 1, "Error: --num-timesteps cannot be less than 1"
    assert args.num_stack >= 1, "Error: --num-stack cannot be less than 1"
    assert args.action_repeat >= 1, "Error: --action-repeat cannot be less than 1"
    assert 0 <= args.port < 65535, "Error: invalid visdom port number {}, ".format(args.port) + \
                                   "port number must be an unsigned 16bit number [0,65535]."
    assert registered_srl[args.srl_model][0] == SRLType.ENVIRONMENT or args.env in all_models, \
        "Error: the environment {} has no srl_model defined in 'srl_models.yaml'. Cannot continue.".format(args.env)
    # check that all the SRL_model can be run on the environment
    if registered_srl[args.srl_model][1] is not None:
        found = False
        for compatible_class in registered_srl[args.srl_model][1]:
            if issubclass(compatible_class, registered_env[args.env][0]):
                found = True
                break
        assert found, "Error: srl_model {}, is not compatible with the {} environment.".format(args.srl_model, args.env)

    assert not(sum([args.simple_continual, args.circular_continual, args.square_continual, args.eight_continual]) \
           > 1 and args.env == "OmnirobotEnv-v0"), \
        "For continual SRL and RL, please provide only one scenario at the time and use OmnirobotEnv-v0 environment !"

    assert not(args.algo == "distillation" and (args.teacher_data_folder == '' or args.continuous_actions is True)), \
        "For performing policy distillation, make sure use specify a valid teacher dataset and discrete actions !"

    ENV_NAME = args.env
    ALGO_NAME = args.algo
    VISDOM_PORT = args.port
    EPISODE_WINDOW = args.episode_window
    MIN_EPISODES_BEFORE_SAVE = args.min_episodes_save
    CROSS_EVAL = args.perform_cross_evaluation_cc
    EPISODE_WINDOW_DISTILLATION_WIN = args.eval_episode_window
    NEW_LR =args.new_lr
    print("EPISODE_WINDOW_DISTILLATION_WIN: ", EPISODE_WINDOW_DISTILLATION_WIN)

    if args.no_vis:
        viz = False

    algo_class, algo_type, action_type = registered_rl[args.algo]
    algo = algo_class()
    ALGO = algo
    
    # if callback frequency needs to be changed
    LOG_INTERVAL = algo.LOG_INTERVAL
    SAVE_INTERVAL = algo.SAVE_INTERVAL

    if not args.continuous_actions and ActionType.DISCRETE not in action_type:
        raise ValueError(args.algo + " does not support discrete actions, please use the '--continuous-actions' " +
                         "(or '-c') flag.")
    if args.continuous_actions and ActionType.CONTINUOUS not in action_type:
        raise ValueError(args.algo + " does not support continuous actions, please remove the '--continuous-actions' " +
                         "(or '-c') flag.")

    env_kwargs["is_discrete"] = not args.continuous_actions

    printGreen("\nAgent = {} \n".format(args.algo))

    env_kwargs["action_repeat"] = args.action_repeat
    # Random init position for button
    env_kwargs["random_target"] = args.random_target

    # If in simple continual scenario, then the target should be initialized randomly.
    if args.simple_continual is True:
        env_kwargs["random_target"] = True

    # Allow up action
    # env_kwargs["force_down"] = False

    # allow multi-view
    env_kwargs['multi_view'] = args.srl_model == "multi_view_srl"
    parser = algo.customArguments(parser)
    args = parser.parse_args()

    args, env_kwargs = configureEnvAndLogFolder(args, env_kwargs, all_models)
    args_dict = filterJSONSerializableObjects(vars(args))
    # Save args
    with open(LOG_DIR + "args.json", "w") as f:
        json.dump(args_dict, f)

    env_class = registered_env[args.env][0]
    # env default kwargs
    default_env_kwargs = {k: v.default
                          for k, v in inspect.signature(env_class.__init__).parameters.items()
                          if v is not None}

    globals_env_param = sys.modules[env_class.__module__].getGlobals()
    ### HACK way to reset image shape !! 
    globals_env_param['RENDER_HEIGHT'] = img_shape[1]
    globals_env_param['RENDER_WIDTH']  = img_shape[2]
    globals_env_param['RELATIVE_POS'] = args.relative_pos

    super_class = registered_env[args.env][1]
    # reccursive search through all the super classes of the asked environment, in order to get all the arguments.
    rec_super_class_lookup = {dict_class: dict_super_class for _, (dict_class, dict_super_class, _, _) in
                              registered_env.items()}
    while super_class != SRLGymEnv:
        assert super_class in rec_super_class_lookup, "Error: could not find super class of {}".format(super_class) + \
                                                      ", are you sure \"registered_env\" is correctly defined?"
        super_env_kwargs = {k: v.default
                            for k, v in inspect.signature(super_class.__init__).parameters.items()
                            if v is not None}
        default_env_kwargs = {**super_env_kwargs, **default_env_kwargs}

        globals_env_param = {**sys.modules[super_class.__module__].getGlobals(), **globals_env_param}

        super_class = rec_super_class_lookup[super_class]

    # Print Variables
    printYellow("Arguments:")
    pprint(args_dict)
    printYellow("Env Globals:")
    pprint(filterJSONSerializableObjects({**globals_env_param, **default_env_kwargs, **env_kwargs}))
    # Save env params
    saveEnvParams(globals_env_param, {**default_env_kwargs, **env_kwargs})
    # Seed tensorflow, python and numpy random generator
    set_global_seeds(args.seed)
    # Augment the number of timesteps (when using mutliprocessing this number is not reached)
    args.num_timesteps = int(1.1 * args.num_timesteps)
    # Get the hyperparameter, if given (Hyperband)
    hyperparams = {param.split(":")[0]: param.split(":")[1] for param in args.hyperparam}
    hyperparams = algo.parserHyperParam(hyperparams)
    
    if args.load_rl_model_path is not None:
        #use a small learning rate
        print("use a small learning rate: {:f}".format(1.0e-4))
        hyperparams["learning_rate"] = lambda f: f * 1.0e-4
        
    # Train the agent
    if args.load_rl_model_path is not None:
        algo.setLoadPath(args.load_rl_model_path)
    algo.train(args, callback, env_kwargs=env_kwargs, train_kwargs=hyperparams)


if __name__ == '__main__':
    main()
