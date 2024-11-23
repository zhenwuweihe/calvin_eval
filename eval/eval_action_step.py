

import argparse
from collections import Counter, defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
from moviepy.editor import ImageSequenceClip
# This is for using the locally installed repo clone when using slurm

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
from calvin_env.envs.play_table_env import get_env
import functools
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import torch.distributed as dist
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000
# NUM_SEQUENCES = 400

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype

def eval_16_action_step_calvin_ddp(args, action_step, dataset_path, eval_log_dir="", debug=False, diverse_inst=False, sequence_i=-1):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type=="diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    conf_dir = Path(args.calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    if diverse_inst:
        with open('./lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        
    eval_log_dir = get_log_dir(eval_log_dir)
    with open('./eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
        
        
    dist.init_process_group(backend='nccl', init_method='env://')
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)
        
    debug = True
    planned_actions = []
    if debug:
        img_queue = []
        
    for initial_state, eval_sequence in eval_sequences:
        start_info = env.get_info()
        for subtask_i, subtask in enumerate(eval_sequence):
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            for i in range(action_step.shape[0]):
                # print(action_step.shape)
                action = action_step[i, ...]
                # print(action.shape)
                action = torch.concat((action[:-1], action[-1:] > 0.5), dim=0)
                action[-1] = (action[-1] - 0.5) * 2 
                action = action.cpu().detach().to(dtype=torch.float16).numpy()
                obs, _, _, current_info = env.step(action)
                if debug:
                    img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
                    img_queue.append(img_copy)
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    if debug:
                        print(colored("success", "green"), end=" ")
                        img_clip = ImageSequenceClip(img_queue, fps=30)
                        img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
                    return True
            break
        break
    
    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False


def save_array_as_png(array, file_name):
    """
    将一个形状为 (height, width, 3) 的 numpy 数组保存为 PNG 文件。
    
    :param array: numpy.ndarray, 必须是 shape == (height, width, 3) 且数据类型为 uint8
    :param file_name: str, 保存的 PNG 文件名，包括路径
    """
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)  # Scale if needed (assuming input is normalized)
    
    # Convert the array to a PIL image
    image = Image.fromarray(array)
    
    # Save the image as a PNG file
    image.save(file_name)

import cv2
def observation2frame(obs, output_resolution=256, griper_view=True):
    # if not griper_view:
    #     pass
    # elif 'rgb_static' in obs and 'rgb_gripper' in obs:
    rgb_static = copy.deepcopy(obs['rgb_obs']['rgb_static'])
    rgb_gripeer = copy.deepcopy(obs['rgb_obs']['rgb_gripper'])

    resized_static = cv2.resize(rgb_static, (output_resolution, output_resolution))
    resized_gripeer = cv2.resize(rgb_gripeer, (output_resolution, output_resolution))
    # Concatenate images along width (axis=1) or height (axis=0)
    # import pdb; pdb.set_trace()
    concatenated_frame = np.concatenate((resized_static, resized_gripeer), axis=1)
    rgb_frame = cv2.cvtColor(concatenated_frame, cv2.COLOR_BGR2RGB)
    return concatenated_frame


def get_first_frame(args, action_step, dataset_path, initial_state, eval_sequence, eval_log_dir="", debug=False, diverse_inst=False, sequence_i=-1):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type=="diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    conf_dir = Path(args.calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    if diverse_inst:
        with open('./lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        
    eval_log_dir = get_log_dir(eval_log_dir)
    # with open('./eval_sequences.json', 'r') as f:
    #     eval_sequences = json.load(f)
    debug = True
    planned_actions = []
    if debug:
        img_queue = []
    # for initial_state, eval_sequence in eval_sequences:
    start_info = env.get_info()
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    concatenated_frame = observation2frame(obs)
    # import pdb; pdb.set_trace()
    img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
    # save_array_as_png(img_copy, 'test.png')
    # return img_copy
    print(f"concatenated_frame : {concatenated_frame.shape}")
    return concatenated_frame, img_copy
    
    
    # import pdb; pdb.set_trace()
        # for i in range(action_step.shape[0]):
        #     # print(action_step.shape)
        #     action = action_step[i, ...]
        #     # print(action.shape)
        #     action = torch.concat((action[:-1], action[-1:] > 0.5), dim=0)
        #     action[-1] = (action[-1] - 0.5) * 2 
        #     action = action.cpu().detach().to(dtype=torch.float16).numpy()
        #     obs, _, _, current_info = env.step(action)
        #     if debug:
        #         img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
        #         img_queue.append(img_copy)
        #     current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        #     if len(current_task_info) > 0:
        #         if debug:
        #             print(colored("success", "green"), end=" ")
        #             img_clip = ImageSequenceClip(img_queue, fps=30)
        #             img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
        #         return True
        # break
        # break
        
def eval_init_state_action_step_calvin_ddp(args, action_step, dataset_path, initial_state, eval_sequence, eval_log_dir="", debug=False, diverse_inst=False, sequence_i=-1):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type=="diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    conf_dir = Path(args.calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    if diverse_inst:
        with open('./lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        
    eval_log_dir = get_log_dir(eval_log_dir)
    # with open('./eval_sequences.json', 'r') as f:
    #     eval_sequences = json.load(f)
        
        
    dist.init_process_group(backend='nccl', init_method='env://')
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    # eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    # if not debug:
    #     eval_sequences = tqdm(eval_sequences, position=0, leave=True)
        
    debug = True
    planned_actions = []
    if debug:
        img_queue = []
        
    # for initial_state, eval_sequence in eval_sequences:
    start_info = env.get_info()
    for subtask_i, subtask in enumerate(eval_sequence):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        for i in range(action_step.shape[0]):
            # print(action_step.shape)
            action = action_step[i, ...]
            # print(action.shape)
            action = torch.concat((action[:-1], action[-1:] > 0.5), dim=0)
            action[-1] = (action[-1] - 0.5) * 2 
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
            obs, _, _, current_info = env.step(action)
            if debug:
                img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
                img_queue.append(img_copy)
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if debug:
                    print(colored("success", "green"), end=" ")
                    img_clip = ImageSequenceClip(img_queue, fps=30)
                    img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
                return True
        break
        # break
    
    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False

        
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=4,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="RobotFlamingo",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_calvin", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--openflamingo_checkpoint", type=str, default="")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        help="path to calvin_dataset",
    )
    parser.add_argument("--loss_multiplier_calvin", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--evaluate_from_checkpoint",
        type=str,
        help="path to checkpoint to evaluate , this should contain model",
        default=None,
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument("--calvin_conf_path", type=str, help="path to calvin configuration file")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(
        "--freeze_embed",
        default=False,
        action="store_true",
        help="freeze the parameters of embedding layer",
    )
    parser.add_argument(
        "--use_gripper",
        default=False,
        action="store_true",
        help="whether to use gripper image as input",
    )
    parser.add_argument(
        "--use_state",
        default=False,
        action="store_true",
        help="whether to use low-dim state as input",
    )
    parser.add_argument(
        "--fusion_mode",
        default="post",
        type=str,
        help="pre or post to fusion multi vision info",
    )
    parser.add_argument("--hist_window", type=int, default=1)  # input history window size for the model
    # history window size when evaluating, for FC head equals to hist_window, for LSTM head means refresh frequency
    parser.add_argument("--eval_hist_size", type=int, default=-1)
    parser.add_argument(
        "--sep_resampler",
        default=False,
        action="store_true",
        help="whether use separate resamplers for third party and gripper camera",
    )
    parser.add_argument("--train_params", type=int, default=-1)
    parser.add_argument('--rgb_pad', type=int, default=-1)
    parser.add_argument('--gripper_pad', type=int, default=-1)
    parser.add_argument('--n_timesteps', type=int, default=150, help="diffusion time steps")
    parser.add_argument(
        "--predict_epsilon",
        default=False,
        action="store_true",
        help="whether diffusion model should predict epsilon",
    )
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument('--head_type', type=str, default="lstm")  # diffusion
    parser.add_argument(
        "--from_scratch",
        default=False,
        action="store_true",
        help="whether to train the model from scratch",
    )
    parser.add_argument("--n_obs_steps", default=6, type=int)
    parser.add_argument("--future_act_len", default=-1, type=int)
    parser.add_argument("--diff_horizon", default=32, type=int)
    parser.add_argument(
        "--last_action",
        default=False,
        action="store_true",
        help="whether using last action as input",
    )
    parser.add_argument(
        "--use_hist",
        default=False,
        action="store_true",
        help="whether using multi-image encoder"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--reset",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--sep_lm_head",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--clip_state",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--convert_rgb",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--diverse_inst",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--tcp_rel",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--replan",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--freeze_sampler",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred_hand",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--no_image_patch",
        default=False,
        action="store_true"
    )
    parser.add_argument("--global_latent", type=int, default=1)
    parser.add_argument("--save_every_iter", type=int, default=-1)
    parser.add_argument("--pad_length", type=int, default=-1)
    # For GPT decoder
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--decoder_type", type=str, default='lstm')

    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--llm_name", type=str, default='llama_9b')
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")


    args = parser.parse_args()
    return args

def main(action_step=None):
    
    args = get_config()
    
    with open('./eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    initial_state, eval_sequence = eval_sequences[0][0], eval_sequences[0][1]
    
    print(f"our task is {eval_sequence[0]}")
    
    concatenated_frame, img_picture = get_first_frame(  
        args=args,
        action_step=action_step,
        dataset_path=args.calvin_dataset, 
        initial_state=initial_state, 
        eval_sequence=eval_sequence
    )
    
    if action_step is None:
        action_step = torch.rand(16, 7)
        
    # eval_16_action_step_calvin_ddp(
    #     args=args,
    #     action_step=action_step,
    #     dataset_path=args.calvin_dataset,
    # )
    
    eval_init_state_action_step_calvin_ddp(
        args=args,
        action_step=action_step,
        dataset_path=args.calvin_dataset, 
        initial_state=initial_state, 
        eval_sequence=eval_sequence
    )
    



if __name__ == '__main__':
    
    main()
