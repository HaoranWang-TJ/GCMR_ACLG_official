import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from algo.models import ControllerActor, ControllerCritic, ManagerActor, ManagerCritic, RndPredictor
from algo.relabel import OffPolicyCorrections, HindsightRelabeling
from algo.utils import AutoLambda
# from algo.utils import RunningMeanStd
from planner.goal_plan import Planner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    return tensor.to(device)


def get_tensor(z):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class Manager(object):
    def __init__(self,
                 state_dim,
                 goal_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 candidate_goals,
                 correction=True,
                 scale=10,
                 actions_norm_reg=0,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 goal_loss_coeff=0,
                 absolute_goal=False,
                 absolute_goal_scale=8.,
                 landmark_loss_coeff=0.,
                 delta=2.0,
                 no_pseudo_landmark=False,
                 automatic_delta_pseudo=False,
                 planner_start_step=50000,
                 planner_cov_sampling='fps',
                 planner_clip_v=-38.,
                 n_landmark_cov=20,
                 planner_initial_sample=1000,
                 planner_goal_thr=-10.,
                 init_opc_delta=0,
                 opc_delta_update_rate=0,
                 correction_type='m-OPC',
                 ):
        self.scale = scale
        self.actor = ManagerActor(state_dim,
                                  goal_dim,
                                  action_dim,
                                  scale=scale,
                                  absolute_goal=absolute_goal,
                                  absolute_goal_scale=absolute_goal_scale)
        self.actor_target = ManagerActor(state_dim,
                                         goal_dim,
                                         action_dim,
                                         scale=scale)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 weight_decay=0.0001)

        self.action_norm_reg = 0

        if torch.cuda.is_available():
            self.actor = self.actor.to(device)
            self.actor_target = self.actor_target.to(device)
            self.critic = self.critic.to(device)
            self.critic_target = self.critic_target.to(device)

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.goal_loss_coeff = goal_loss_coeff
        self.absolute_goal = absolute_goal

        self.landmark_loss_coeff = landmark_loss_coeff
        self.delta = delta
        self.device = device
        self.planner = None
        self.no_pseudo_landmark = no_pseudo_landmark

        self.automatic_delta_pseudo = automatic_delta_pseudo
        if self.automatic_delta_pseudo:
            self.delta = 0.

        self.planner_start_step = planner_start_step
        self.planner_cov_sampling = planner_cov_sampling
        self.planner_clip_v = planner_clip_v
        self.n_landmark_cov = n_landmark_cov
        self.planner_initial_sample = planner_initial_sample
        self.planner_goal_thr = planner_goal_thr

        self.opc_delta_f = AutoLambda(init_opc_delta, opc_delta_update_rate)
        self.correction_type = correction_type

    def init_planner(self):
        self.planner = Planner(landmark_cov_sampling=self.planner_cov_sampling,
                               clip_v=self.planner_clip_v,
                               n_landmark_cov=self.n_landmark_cov,
                               initial_sample=self.planner_initial_sample,
                               goal_thr=self.planner_goal_thr)

    def set_delta(self, data, alpha=0.9):
        assert self.automatic_delta_pseudo
        if self.delta == 0:
            self.delta = data
        else:
            self.delta = alpha * data + (1 - alpha) * self.delta

    def set_eval(self):
        self.actor.set_eval()
        self.actor_target.set_eval()

    def set_train(self):
        self.actor.set_train()
        self.actor_target.set_train()

    def sample_goal(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        if to_numpy:
            return self.actor(state, goal).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, goal).squeeze()

    def value_estimate(self, state, goal, subgoal):
        return self.critic(state, goal, subgoal)

    def get_pseudo_landmark(self, ag, planned_ld):
        direction = planned_ld - ag
        norm_direction = F.normalize(direction)
        scaled_norm_direction = norm_direction * self.delta
        pseudo_landmarks = ag.clone()
        pseudo_landmarks[~torch.isnan(scaled_norm_direction)] = pseudo_landmarks[~torch.isnan(scaled_norm_direction)] +\
                                                            scaled_norm_direction[~torch.isnan(scaled_norm_direction)]

        scaled_norm_direction[scaled_norm_direction != scaled_norm_direction] = 0
        return pseudo_landmarks, scaled_norm_direction.mean(dim=0)

    def actor_loss(self, state, achieved_goal, goal, a_net, r_margin, selected_landmark=None, no_pseudo_landmark=False):
        actions = self.actor(state, goal)
        eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        if a_net is None:
            return eval + norm  # HIRO

        scaled_norm_direction = var(torch.FloatTensor([0.] * self.action_dim))
        gen_subgoal = actions if self.absolute_goal else achieved_goal + actions
        goal_loss = torch.clamp(F.pairwise_distance(a_net(achieved_goal), a_net(gen_subgoal)) - r_margin, min=0.).mean()
        if selected_landmark is None:
            return eval + norm, goal_loss, None, scaled_norm_direction  # HRAC

        if no_pseudo_landmark:
            selected_landmark[selected_landmark == float('inf')] = achieved_goal[selected_landmark == float('inf')]
            batch_landmarks = selected_landmark.clone()
        else:
            batch_landmarks, scaled_norm_direction = self.get_pseudo_landmark(achieved_goal, selected_landmark)
        ld_loss = torch.clamp(F.pairwise_distance(a_net(batch_landmarks), a_net(gen_subgoal)) - r_margin, min=0.).mean()

        follow_loss = F.mse_loss(batch_landmarks, gen_subgoal).mean()

        return eval + norm, goal_loss, ld_loss, follow_loss, scaled_norm_direction  # HIGL

    def select_subgoal(self, state, goal, to_numpy=True):
        if not torch.is_tensor(state):
            state = get_tensor(state)
        if not torch.is_tensor(goal):
            goal = get_tensor(goal)

        if to_numpy:
            return self.actor(state, goal).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, goal).squeeze()

    def goal_relabeling(self, controller_policy, batch_size, subgoals, x_seq, a_seq, ag_seq, goals, fkm_obj=None, exp_w=1.0):
        if self.correction_type == 'm-OPC':
            opc_obj = OffPolicyCorrections(self.absolute_goal, controller_policy, batch_size, subgoals.copy(), x_seq, a_seq, ag_seq, self.candidate_goals, self.scale, self.action_dim, fkm_obj, exp_w)
            relabeled_goals = opc_obj.get_corrected_goals()
        elif self.correction_type == 'OSP':
            hr_obj = HindsightRelabeling(self.absolute_goal, self, controller_policy, batch_size, subgoals.copy(), x_seq, ag_seq, goals, self.scale, self.action_dim, fkm_obj)
            relabeled_goals = hr_obj.get_relabeled_goals()
        elif self.correction_type == 'OPC':
            opc_obj = OffPolicyCorrections(self.absolute_goal, controller_policy, batch_size, subgoals.copy(), x_seq, a_seq, ag_seq, self.candidate_goals, self.scale, self.action_dim, None, exp_w)
            relabeled_goals = opc_obj.get_corrected_goals()
        elif self.correction_type == 'HAC':
            hr_obj = HindsightRelabeling(self.absolute_goal, self, controller_policy, batch_size, subgoals.copy(), x_seq, ag_seq, goals, self.scale, self.action_dim, None)
            relabeled_goals = hr_obj.get_relabeled_goals()
        else:
            return subgoals

        if not self.opc_delta_f.enable:
            return np.vstack(relabeled_goals)

        vec_norm = lambda x: x / (np.linalg.norm(x) + 1e-7)
        sg_direction = vec_norm(relabeled_goals - subgoals)
        if self.opc_delta_f.is_dynamic:
            self.opc_delta_f.update(np.linalg.norm(relabeled_goals - subgoals, axis=1).mean())
        soft_subgoals = subgoals + self.opc_delta_f.value * sg_direction
        soft_subgoals = soft_subgoals.clip(-self.scale[:self.action_dim], self.scale[:self.action_dim])
        return soft_subgoals

    def train(self,
              algo,
              controller_policy,
              replay_buffer,
              controller_replay_buffer,
              iterations,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              a_net=None,
              r_margin=None,
              total_timesteps=None,
              novelty_pq=None,
              fkm_obj=None,
              exp_w=1.0,
              ):
        self.manager_buffer = replay_buffer
        avg_act_loss, avg_crit_loss, avg_goal_loss, avg_ld_loss, avg_floss, avg_norm_sel = 0., 0., 0., 0., 0., 0.
        avg_scaled_norm_direction = get_tensor(np.array([0.] * self.action_dim)).squeeze()

        if algo in ['higl', 'aclg'] and self.planner is None and total_timesteps >= self.planner_start_step:
            self.init_planner()

        for it in range(iterations):
            # Sample replay buffer
            x, y, ag, ag_next, g, sgorig, r, d, xobs_seq, a_seq, ag_seq = replay_buffer.sample(batch_size)

            if self.correction:
                sg = self.goal_relabeling(controller_policy, batch_size, sgorig, xobs_seq, a_seq, ag_seq, g, fkm_obj=fkm_obj, exp_w=exp_w)
            else:
                sg = sgorig

            state = get_tensor(x)
            next_state = get_tensor(y)
            achieved_goal = get_tensor(ag)
            goal = get_tensor(g)
            subgoal = get_tensor(np.array(sg))

            reward = get_tensor(r)
            done = get_tensor(1 - d)

            noise = torch.FloatTensor(sgorig).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(state, goal, subgoal)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) + \
                          self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if algo == "hiro":
                actor_loss = self.actor_loss(state, achieved_goal, goal, a_net, r_margin)

            elif algo == "hrac":
                assert a_net is not None

                actor_loss, goal_loss, _, _ = \
                    self.actor_loss(state, achieved_goal, goal, a_net, r_margin, selected_landmark=None)
                actor_loss = actor_loss + self.goal_loss_coeff * goal_loss
                avg_goal_loss += goal_loss

            elif algo in ['higl', 'aclg']:
                assert a_net is not None

                if self.planner is None:  # If planner is not ready
                    selected_landmark = torch.ones(len(state), self.action_dim).to(device)
                    selected_landmark *= float("inf")  # Build dummy selected landmark
                else:  # Select a landmark by a planner
                    selected_landmark = self.planner(cur_obs=x,
                                                     cur_ag=ag,
                                                     final_goal=g,
                                                     agent=controller_policy,
                                                     replay_buffer=controller_replay_buffer,
                                                     novelty_pq=novelty_pq)
                    if self.automatic_delta_pseudo:
                        ag2sel = np.linalg.norm(selected_landmark.cpu().numpy() - ag, axis=1).mean()
                        self.set_delta(ag2sel)

                actor_loss, goal_loss, ld_loss, follow_loss, scaled_norm_direction = self.actor_loss(state, achieved_goal, goal,
                                                                                        a_net, r_margin,
                                                                                        selected_landmark,
                                                                                        self.no_pseudo_landmark)
                if algo == "higl":
                    actor_loss = actor_loss + self.landmark_loss_coeff * ld_loss
                elif algo == "aclg":
                    actor_loss = actor_loss + self.goal_loss_coeff * goal_loss + self.landmark_loss_coeff * follow_loss
                else:
                    raise NotImplementedError

                avg_goal_loss += goal_loss
                avg_ld_loss += ld_loss
                avg_floss += follow_loss
                avg_scaled_norm_direction += scaled_norm_direction
            else:
                raise NotImplementedError

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, \
               avg_crit_loss / iterations, \
               avg_goal_loss / iterations, \
               avg_ld_loss / iterations,\
               avg_floss / iterations,\
               avg_scaled_norm_direction / iterations

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, dir, env_name, algo, version, seed):
        torch.save(self.actor.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerActor.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerCritic.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.actor_target.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerActorTarget.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic_target.state_dict(),
                   "{}/{}_{}_{}_{}_ManagerCriticTarget.pth".format(dir, env_name, algo, version, seed))
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo, version, seed):
        self.actor.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerActor.pth".format(dir, env_name, algo, version, seed)))
        self.critic.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerCritic.pth".format(dir, env_name, algo, version, seed)))
        self.actor_target.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerActorTarget.pth".format(dir, env_name, algo, version, seed)))
        self.critic_target.load_state_dict(
            torch.load("{}/{}_{}_{}_{}_ManagerCriticTarget.pth".format(dir, env_name, algo, version, seed)))
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo)))


class Controller(object):
    def __init__(self,
                 state_dim,
                 goal_dim,
                 action_dim,
                 max_action,
                 actor_lr,
                 critic_lr,
                 no_xy=True,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 absolute_goal=False,
                 man_policy_noise=0.2,
                 man_policy_noise_clip=0.5,
    ):
        self.actor = ControllerActor(state_dim, goal_dim, action_dim, scale=max_action)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim, scale=max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.0001)

        self.no_xy = no_xy

        self.subgoal_transition = self.subgoal_transition

        if torch.cuda.is_available():
            self.actor = self.actor.to(device)
            self.actor_target = self.actor_target.to(device)
            self.critic = self.critic.to(device)
            self.critic_target = self.critic_target.to(device)

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.absolute_goal = absolute_goal

        self.device = device

        self._auto_upperbounded_k = 0.
        self.man_policy_noise = man_policy_noise
        self.man_policy_noise_clip = man_policy_noise_clip
        self.osrp_interval = 0
        self.mgp_interval = 0

    def clean_obs(self, state, dims=2):
        if self.no_xy:
            with torch.no_grad():
                mask = torch.ones_like(state)
                if len(state.shape) == 3:
                    mask[:, :, :dims] = 0
                elif len(state.shape) == 2:
                    mask[:, :dims] = 0
                elif len(state.shape) == 1:
                    mask[:dims] = 0

                return state*mask
        else:
            return state

    def select_action(self, state, sg, to_numpy=True):
        if not torch.is_tensor(state):
            state = get_tensor(state)
        if not torch.is_tensor(sg):
            sg = get_tensor(sg)
        state = self.clean_obs(state)

        if to_numpy:
            return self.actor(state, sg).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, sg).squeeze()

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def _get_osrp_loss(self, fkm_obj, manage_replay_buffer, manage_actor, manage_critic, batch_size, sg_scale):
        # x, g, sg
        # Sample replay buffer
        _curr_state, _, _, _, _dg, _sgorig, _, _, _, _, _ = manage_replay_buffer.sample(batch_size)
        _curr_state = _curr_state.repeat(100, 0)
        _dg = _dg.repeat(10, 0)
        np.random.shuffle(_dg)
        _dg = _dg.repeat(10, 0)
        _curr_state = get_tensor(_curr_state)
        _dg = get_tensor(_dg)

        if sg_scale is None:
            _sgorig = _sgorig.repeat(100, 0)
            _sgorig = _sgorig + np.random.rand(*_sgorig.shape) * 2 - 1
        else:
            _sgorig = np.random.normal(loc=manage_actor(_curr_state, _dg).cpu().data.numpy(),
                                        scale=.5*sg_scale)
            _sgorig = _sgorig.clip(-sg_scale, sg_scale)
        _sgorig = get_tensor(_sgorig)

        _action = self.actor(self.clean_obs(_curr_state), _sgorig)
        _state_delta = fkm_obj(_curr_state, _action, batch_size=batch_size)
        _next_s = _curr_state + _state_delta

        if fkm_obj.scaler.obs_max is not None and fkm_obj.scaler.obs_min is not None:
            _next_s = _next_s.clamp(get_tensor(fkm_obj.scaler.obs_min), get_tensor(fkm_obj.scaler.obs_max))

        if self.absolute_goal:
            _new_sg = _sgorig
        else:
            _new_sg = _sgorig + _curr_state[:, :_sgorig.size(1)] - _next_s[:, :_sgorig.size(1)]
        _target_Q1, _target_Q2 = manage_critic(_next_s, _dg, _new_sg)
        _target_Q = torch.min(_target_Q1, _target_Q2)
        actor_osrp_loss = - _target_Q.mean()

        _new_sg_man = manage_actor(_next_s, _dg)
        _new_sg_man_noise = np.random.normal(loc=np.zeros(_new_sg_man.size()), scale=self.man_policy_noise)
        _new_sg_man_noise = _new_sg_man_noise.clip(-self.man_policy_noise_clip, self.man_policy_noise_clip)
        _new_sg_man = _new_sg_man + get_tensor(_new_sg_man_noise)
        _new_sg_man = _new_sg_man.clamp(get_tensor(-sg_scale), get_tensor(sg_scale))
        _target_Q1_man, _target_Q2_man = manage_critic(_next_s, _dg, _new_sg_man)
        _target_Q_man = torch.min(_target_Q1_man, _target_Q2_man)
        actor_osrp_man_loss = - _target_Q_man.mean()

        return 0.5 * (actor_osrp_loss + actor_osrp_man_loss)

    def actor_loss(self, state, sg, fkm_obj=None, state_f=None \
                   , osrp_lambda=.0, manage_replay_buffer=None, manage_actor=None, manage_critic=None, batch_size=256, sg_scale=None):
        actions = self.actor(state, sg)
        act_Q = self.critic.Q1(state, sg, actions)
        act_loss = -1 * act_Q.mean()
        osrp_loss = .0

        if fkm_obj is not None and fkm_obj.trained and state_f is not None and manage_actor is not None and manage_critic is not None:
            osrp_loss = osrp_lambda * self._get_osrp_loss(fkm_obj, manage_replay_buffer, manage_actor, manage_critic, batch_size=batch_size, sg_scale=sg_scale)

        return act_loss, osrp_loss

    def subgoal_transition(self, achieved_goal, subgoal, next_achieved_goal):
        if self.absolute_goal:
            return subgoal
        else:
            if len(achieved_goal.shape) == 1:  # check if batched
                return achieved_goal + subgoal - next_achieved_goal
            else:
                return achieved_goal[:, ] + subgoal - next_achieved_goal[:, ]

    def multi_subgoal_transition(self, achieved_goal, subgoal):
        subgoals = (subgoal + achieved_goal[:, 0, ])[:, None] - achieved_goal[:, :, ]
        return subgoals

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, fkm_obj=None, mgp_lambda=.0, \
              osrp_lambda=.0, manage_replay_buffer=None, manage_actor=None, manage_critic=None, sg_scale=None):

        avg_act_loss = dict({'avg_act_loss': 0., 'avg_act_osrp_loss': 0.})
        avg_crit_loss = dict({'avg_crit_loss': 0., 'avg_mgp_loss': 0.})

        use_mgp = mgp_lambda > 0 and fkm_obj is not None and fkm_obj.trained
        extend_train_scale = 5
        if use_mgp:
            iterations = iterations * extend_train_scale

        for it in range(iterations):
            # Sample replay buffer
            x, y, ag, ag_next, sg, u, r, d, _, _, _ = replay_buffer.sample(batch_size)

            next_g = get_tensor(self.subgoal_transition(ag, sg, ag_next))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            next_state = self.clean_obs(get_tensor(y))

            noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_g) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, next_g, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, sg, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                          self.criterion(current_Q2, target_Q_no_grad)

            # critic GP
            if use_mgp and self.mgp_interval >= 5:
                _state_rep = state.clone().detach().repeat(16, 1).requires_grad_(True)
                _sg = sg.clone().detach().repeat(16, 1).requires_grad_(True)
                _random_action = torch.rand(
                    size=self.actor(_state_rep, _sg).size(),
                    requires_grad=True) * 2 - 1.0
                _random_action= _random_action.to(self.device)
                _current_Q1, _current_Q2 = self.critic(_state_rep, _sg, _random_action)
                grad_q1_wrt_random_action = torch.autograd.grad(
                    outputs=_current_Q1.sum(),
                    inputs =_random_action,
                    create_graph=True)[0].norm(p=2, dim=-1)
                grad_q2_wrt_random_action = torch.autograd.grad(
                    outputs=_current_Q2.sum(),
                    inputs =_random_action,
                    create_graph=True)[0].norm(p=2, dim=-1)

                _global_state = get_tensor(x).clone().detach().repeat(16, 1).requires_grad_(True)
                _state_delta = fkm_obj(_global_state, _random_action, batch_size=len(_global_state))
                if not self.absolute_goal:
                    delta_rwd = torch.autograd.grad(
                        outputs=(_state_delta[:, :_sg.size(1)]-_sg).norm(p=2, dim=-1).sum(),
                        inputs=_random_action,
                        create_graph=False)[0].norm(p=2, dim=-1).max()
                else:
                    delta_rwd = torch.autograd.grad(
                        outputs=(_global_state[:, :_sg.size(1)] + _state_delta[:, :_sg.size(1)]-_sg).norm(p=2, dim=-1).sum(),
                        inputs=_random_action,
                        create_graph=False)[0].norm(p=2, dim=-1).max()

                self._auto_upperbounded_k = np.math.sqrt(_random_action.size(1)) / (1 - discount) * delta_rwd.detach() / len(_global_state)

                grad_q_wrt_random_action = F.relu(grad_q1_wrt_random_action - self._auto_upperbounded_k) **2 +\
                        F.relu(grad_q2_wrt_random_action - self._auto_upperbounded_k) **2
                mgp_loss = mgp_lambda * grad_q_wrt_random_action.mean()
                avg_crit_loss['avg_mgp_loss'] += mgp_loss
                critic_loss += mgp_loss

                self.mgp_interval = 0
            else:
                self.mgp_interval += 1

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            avg_crit_loss['avg_crit_loss'] += critic_loss
            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if it % extend_train_scale == extend_train_scale - 1:
                # Compute actor loss
                act_loss, osrp_loss = self.actor_loss(state, sg, fkm_obj=fkm_obj , state_f=get_tensor(x)
                                            , osrp_lambda=osrp_lambda if self.osrp_interval >= 10 else 0.
                                            , manage_replay_buffer=manage_replay_buffer, manage_actor=manage_actor if self.osrp_interval >= 10 else None
                                            , manage_critic=manage_critic if self.osrp_interval >= 10 else None, batch_size=batch_size, sg_scale=sg_scale)
                if (osrp_loss - .0) > 1e-7:
                    self.osrp_interval = 0
                else:
                    self.osrp_interval += 1

                avg_act_loss['avg_act_osrp_loss'] += osrp_loss
                actor_loss = act_loss + osrp_loss

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                avg_act_loss['avg_act_loss'] += actor_loss
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for key in avg_act_loss:
            avg_act_loss[key] /= iterations / (extend_train_scale if use_mgp else 1)
        for key in avg_crit_loss:
            avg_crit_loss[key] /= iterations

        return avg_act_loss, avg_crit_loss

    def save(self, dir, env_name, algo, version, seed):
        torch.save(self.actor.state_dict(), "{}/{}_{}_{}_{}_ControllerActor.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic.state_dict(), "{}/{}_{}_{}_{}_ControllerCritic.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.actor_target.state_dict(), "{}/{}_{}_{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo, version, seed))
        torch.save(self.critic_target.state_dict(), "{}/{}_{}_{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo, version, seed))

    def load(self, dir, env_name, algo, version, seed):
        self.actor.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerActor.pth".format(dir, env_name, algo, version, seed)))
        self.critic.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerCritic.pth".format(dir, env_name, algo, version, seed)))
        self.actor_target.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo, version, seed)))
        self.critic_target.load_state_dict(torch.load("{}/{}_{}_{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo, version, seed)))

    def pairwise_value(self, obs, ag, goal):
        assert ag.shape[0] == goal.shape[0]
        with torch.no_grad():
            if not self.absolute_goal:
                relative_goal = goal - ag
                cleaned_obs = self.clean_obs(obs)
                actions = self.actor(cleaned_obs, relative_goal)
                dist1, dist2 = self.critic(cleaned_obs, relative_goal, actions)
                dist = torch.min(dist1, dist2)
                return dist.squeeze(-1)
            else:
                cleaned_obs = self.clean_obs(obs)
                actions = self.actor(cleaned_obs, goal)
                dist1, dist2 = self.critic(cleaned_obs, goal, actions)
                dist = torch.min(dist1, dist2)
                return dist.squeeze(-1)


class RandomNetworkDistillation(object):
    def __init__(self, input_dim, output_dim, lr, use_ag_as_input=False):
        self.predictor = RndPredictor(input_dim, output_dim)
        self.predictor_target = RndPredictor(input_dim, output_dim)

        if torch.cuda.is_available():
            self.predictor = self.predictor.to(device)
            self.predictor_target = self.predictor_target.to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.use_ag_as_input = use_ag_as_input

    def get_novelty(self, obs):
        obs = get_tensor(obs)
        with torch.no_grad():
            target_feature = self.predictor_target(obs)
            feature = self.predictor(obs)
            novelty = (feature - target_feature).pow(2).sum(1).unsqueeze(1) / 2
        return novelty

    def train(self, replay_buffer, iterations, batch_size=100):
        for it in range(iterations):
            # Sample replay buffer
            x, _, ag, _, _, _, _, _, _, _, _ = replay_buffer.sample(batch_size)

            input = x if not self.use_ag_as_input else ag
            input = get_tensor(input)

            with torch.no_grad():
                target_feature = self.predictor_target(input)
            feature = self.predictor(input)
            loss = (feature - target_feature).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss
