ENV=$1
TIMESTEPS=$2
GPU=$3
SEED=$4

# The hyperparameters associated with method A are marked with backslash (\\**\\)
if [[ ${ENV} == "Pusher-v0" ]];
then
    glc=0
    delta=2
    ld_num=20
else
    glc=20
    delta=3
    ld_num=60
fi

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--absolute_goal \
--delta ${delta} \
--env_name ${ENV} \
--reward_shaping "sparse" \
--algo aclg \
\
\
--correction_type m-OPC \
--use_model_based_rollout \
--fkm_hidden_size 256 \
--fkm_hidden_layer_num 3 \
--fkm_network_num 5 \
--fkm_batch_size 512 \
--fkm_lr 0.005 \
--fkm_obj_start_step 10000 \
--train_fkm_freq 500 \
--osp_delta 10 \
--osp_delta_update_rate 0 \
--rollout_exp_w 0.95 \
--ctrl_mgp_lambda 1.0  \
--ctrl_osrp_lambda 0.0005 \
--ctrl_gcmr_start_step 10000 \
\
\
--goal_loss_coeff ${glc} \
--landmark_loss_coeff 1 \
--seed ${SEED} \
--max_timesteps  ${TIMESTEPS} \
--manager_propose_freq 5 \
--landmark_sampling fps \
--n_landmark_coverage ${ld_num} \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty ${ld_num} \
--ctrl_noise_sigma 0.1 \
--man_noise_sigma 0.2 \
--train_ctrl_policy_noise 0.1 \
--train_man_policy_noise 0.2 \
--ctrl_rew_scale 0.1 \
--r_margin_pos 0.01 \
--r_margin_neg 0.012 \
--close_thr 0.02 \
--clip_v -15 \
--goal_thr -5 \
--version "sparse_gcmr"
