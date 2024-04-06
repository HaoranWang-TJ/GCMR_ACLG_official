import numpy as np

class OffPolicyCorrections(object):
    def __init__(self, absolute_goal:bool, controller_policy, batch_size, subgoals, obs, acts, ags, candidate_num
                 , subgoal_scale, subgoal_dim, fkm_obj, exp_w=0.0):
        self.absolute_goal = absolute_goal
        self.controller_policy = controller_policy
        self.batch_size = batch_size
        self.subgoals = subgoals
        self.obs = obs
        self.acts = acts
        self.ags = ags
        self.candidate_num = candidate_num
        self.subgoal_scale = subgoal_scale
        self.subgoal_dim = subgoal_dim

        self.fkm_obj = fkm_obj

        # pipeline
        self.candidates = self.get_candidates()
        self.rollout_error = self.get_rollout_error(exp_w)
        
    def get_candidates(self):
        first_ag = [x[0] for x in self.ags]
        last_ag = [x[-1] for x in self.ags]

        # Shape: (batchsz, 1, subgoal_dim)
        diff_goal = (np.array(last_ag) - np.array(first_ag))[:, np.newaxis, ]

        # Shape: (batchsz, 1, subgoal_dim)
        original_goal = np.array(self.subgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal,
                                        scale=.5*self.subgoal_scale[None, None, :self.subgoal_dim],
                                        size=(self.batch_size, self.candidate_num, original_goal.shape[-1]))
        if self.absolute_goal:
            random_goals = np.array(first_ag)[:, np.newaxis, ] + random_goals
        random_goals = random_goals.clip(-self.subgoal_scale[:self.subgoal_dim], self.subgoal_scale[:self.subgoal_dim])

        # Shape: (batchsz, 10, subgoal_dim)
        if self.absolute_goal:
            candidates = np.concatenate([original_goal, np.array(last_ag)[:, np.newaxis, ], random_goals], axis=1)
        else:
            candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)

        return candidates

    def get_rollout_error(self, exp_w=0.0):
        x_seq = np.array(self.obs)[:, :-1, :]
        a_seq = np.array(self.acts)
        seq_len = len(x_seq[0])

        # For ease
        action_dim = a_seq[0][0].shape
        obs_dim = x_seq[0][0].shape
        ncands = self.candidates.shape[1]

        def _hiro():
            new_batch_sz = seq_len * self.batch_size
            _true_actions = a_seq.reshape((new_batch_sz,) + action_dim)
            _observations = x_seq.reshape((new_batch_sz,) + obs_dim)
            _goal_shape = (new_batch_sz, self.subgoal_dim)
            
            _policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

            for c in range(ncands):
                if self.absolute_goal:
                    _candidate = self.candidates[:, c].repeat(seq_len, axis=0)
                else:
                    _candidate = self.controller_policy.multi_subgoal_transition(np.array(self.ags)[:, :-1, :], self.candidates[:, c])
                    _candidate = _candidate.reshape(*_goal_shape)
                _policy_actions[c] = self.controller_policy.select_action(_observations, _candidate)

            difference = (_policy_actions - _true_actions)

            difference = np.where(difference != -np.inf, difference, 0)
            # ncands * batch_size * seq_len * dim -> batch_size * ncands * seq_len * dim
            difference = difference.reshape((ncands, self.batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)
            return difference

        def _m_hiro():
            # model-based rollout
            _policy_actions = np.zeros((self.batch_size * ncands, seq_len) + action_dim)
            # batch_size * ncands * action_dim
            _curr_state = np.repeat(np.array(x_seq)[:, 0:1, :], ncands, axis=1)
            _curr_state = _curr_state.reshape((self.batch_size * ncands, -1))
            # batch_size * ncands * self.action_dim
            _candidate = self.candidates.reshape((self.batch_size * ncands, self.subgoal_dim))
            for t in range(seq_len):
                _policy_actions[:, t] = self.controller_policy.select_action(_curr_state, _candidate, to_numpy=True)
                _state_delta = self.fkm_obj.get_next_state(_curr_state, _policy_actions[:, t])
                _next_state = _curr_state + _state_delta
                if self.fkm_obj.scaler.obs_max is not None and self.fkm_obj.scaler.obs_min is not None:
                    _next_state = _next_state.clip(self.fkm_obj.scaler.obs_min, self.fkm_obj.scaler.obs_max)
                _next_state = (1 - exp_w**(t+1)) * _next_state + exp_w**(t+1) * np.repeat(
                    np.array(self.obs)[:, t+1:t+2, :], ncands, axis=1).reshape((self.batch_size * ncands, -1))
                # new state
                if not self.absolute_goal:
                    _candidate = (_candidate + _curr_state[:, :self.subgoal_dim]) - _next_state[:, :self.subgoal_dim]
                    _candidate = _candidate.clip(-self.subgoal_scale[:self.subgoal_dim], self.subgoal_scale[:self.subgoal_dim])
                _curr_state = _next_state

            # batch_size, ncands, seq_len, action_dim
            difference = _policy_actions.reshape((self.batch_size, ncands, seq_len) + action_dim) - a_seq.reshape((self.batch_size, 1, seq_len) + action_dim)
            difference = np.where(difference != -np.inf, difference, 0)
            return difference
    
        if self.fkm_obj is not None and self.fkm_obj.trained:
            difference = _m_hiro()
        else:
            difference = _hiro()
        return difference
    
    def get_corrected_goals(self):
        difference = self.rollout_error

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        hit_candidates = self.candidates[np.arange(self.batch_size), max_indices]
        
        return hit_candidates
    

class HindsightRelabeling(object):
    def __init__(self, absolute_goal:bool, manager_policy, controller_policy, batch_size, subgoals, obs, ac_g, goals, subgoal_scale, subgoal_dim, fkm_obj):
        self.absolute_goal = absolute_goal
        self.manager_policy = manager_policy
        self.controller_policy = controller_policy
        self.batch_size = batch_size
        self.subgoals = subgoals
        self.obs = obs
        self.ac_g = ac_g
        self.goals = goals
        self.subgoal_scale = subgoal_scale
        self.subgoal_dim = subgoal_dim

        self.fkm_obj = fkm_obj


    def get_relabeled_goals(self):
        x_seq = np.array(self.obs)[:, :-1, :]
        seq_len = len(x_seq[0])

        def _hac():
            last_ag = [x[-1] for x in self.ac_g]
            return last_ag

        def _fgi():
            # model-based rollout
            # batch_size * ncands * action_dim
            _curr_state = np.array(x_seq)[:, 0, :]
            _subgoal = self.manager_policy.select_subgoal(_curr_state, self.goals, to_numpy=True)

            for t in range(seq_len):
                _act = self.controller_policy.select_action(_curr_state, _subgoal, to_numpy=True)
                _state_delta = self.fkm_obj.get_next_state(_curr_state, _act)
                _next_state = _curr_state + _state_delta
                if self.fkm_obj.scaler.obs_max is not None and self.fkm_obj.scaler.obs_min is not None:
                    _next_state = _next_state.clip(self.fkm_obj.scaler.obs_min, self.fkm_obj.scaler.obs_max)
                # new state
                if not self.absolute_goal:
                    _subgoal = (_subgoal + _curr_state[:, :self.subgoal_dim]) - _next_state[:, :self.subgoal_dim]
                    _subgoal = _subgoal.clip(-self.subgoal_scale[:self.subgoal_dim], self.subgoal_scale[:self.subgoal_dim])
                _curr_state = _next_state
            
            return _curr_state[:, :self.subgoal_dim]
    
        if self.fkm_obj is not None and self.fkm_obj.trained:
            # the Foresight Goal Inference (FGI), please refer to "MapGo: Model-Assisted Policy Optimization for Goal-Oriented Tasks"
            relabeled_goals = _fgi()
        else:
            # refer to "LEARNING MULTI-LEVEL HIERARCHIES WITH HINDSIGHT"
            relabeled_goals = _hac()
        return relabeled_goals