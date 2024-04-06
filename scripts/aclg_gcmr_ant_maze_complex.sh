REWARD_SHAPING=$1
TIMESTEPS=$2
GPU=$3
SEED=$4

# The hyperparameters associated with method A are marked with backslash (\\**\\)

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntMazeComplex-v2" \
--reward_shaping ${REWARD_SHAPING} \
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
--fkm_obj_start_step 20000 \
--train_fkm_freq 2000 \
--osp_delta 30 \
--osp_delta_update_rate 0 \
--rollout_exp_w 0.95 \
--ctrl_mgp_lambda 1.0 \
--ctrl_osrp_lambda 0.0005 \
--ctrl_gcmr_start_step 20000 \
\
\
--version "${REWARD_SHAPING}_gcmr" \
--goal_loss_coeff 20 \
--landmark_loss_coeff 1 \
--delta 3.0 \
--seed ${SEED} \
--max_timesteps ${TIMESTEPS} \
--landmark_sampling fps \
--n_landmark_coverage 200 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 60