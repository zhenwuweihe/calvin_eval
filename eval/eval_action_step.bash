# export PATH=$PATH:/share/dmh/RoboFlamingo/robot_flamingo
# export PYTHONPATH=$PYTHONPATH:/share/dmh/RoboFlamingo/robot_flamingo
# export PATH=$PATH:/share/dmh/RoboFlamingo/open_flamingo
# export PYTHONPATH=$PYTHONPATH:/share/dmh/RoboFlamingo/open_flamingo
# export PATH=$PATH:/share/dmh/RoboFlamingo/calvin/calvin_env
# export PYTHONPATH=$PYTHONPATH:/share/dmh/RoboFlamingo/calvin/calvin_env
# export PATH=$PATH:/share/dmh
# export PYTHONPATH=$PYTHONPATH:/share/dmh
# export PATH=$PATH:/share/dmh/robomamba
# export PYTHONPATH=$PYTHONPATH:/share/dmh/robomamba
export PATH=$PATH:../calvin_models
export PYTHONPATH=$PYTHONPATH:../calvin_models

export PATH=$PATH:../calvin_env
export PYTHONPATH=$PYTHONPATH:../calvin_env



calvin_dataset_path='/share/dmh/cobra/cobra/cobra/dataset/task_ABCD_D'
# calvin_conf_path
calvin_conf_path="../calvin_models/conf"
# language model path
lm_path="facebook/opt-30b"
# tokenizer path
tokenizer_path="facebook/opt-30b"

evaluate_from_checkpoint=None
log_file='./logs_new/evaluate_result.log'
use_gripper=False
use_state=False
fusion_mode='post'
window_size=12
export MESA_GL_VERSION_OVERRIDE=4.1
echo logging to ${log_file}
node_num=1
export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6066 eval_action_step.py \
    --precision fp32 \
    --use_gripper \
    --use_state \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --workers 1 
    # > ${log_file} 2>&1