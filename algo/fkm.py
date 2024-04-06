import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from sklearn.model_selection import train_test_split
import itertools

from algo.higl import device, get_tensor

if device.type == 'cpu':
    torch.set_default_tensor_type(torch.FloatTensor)
else:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

__all__ = [
    "FKMInterface"
]


def safe_convert_tensor(x):
    if not torch.is_tensor(x):
        return get_tensor(x)
    return x


class StandardScaler:
    # https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
    def __init__(self, mean=None, std=None, state_dim=-1, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.state_dim = state_dim
        self.state_min = None
        self.state_max = None
        self.epsilon = epsilon

    def fit(self, values):
        with torch.no_grad():
            values = safe_convert_tensor(values)
            self.mean = torch.mean(values, dim=0, keepdim=True)
            self.std = torch.std(values, dim=0, keepdim=True)
            _state_min, _ = torch.min(values, dim=0, keepdim=True)
            _state_max, _ = torch.max(values, dim=0, keepdim=True)
            if self.state_min is not None:
                self.state_min = self.state_min - 1.5 * torch.abs(_state_min - self.state_min)
            else:
                self.state_min = _state_min - 0.5 * torch.abs(_state_min)
            if self.state_max is not None:
                self.state_max = self.state_max + 1.5 * torch.abs(_state_max - self.state_max)
            else:
                self.state_max = _state_max + 0.5 * torch.abs(_state_max)

    def transform(self, values):
        values = safe_convert_tensor(values)
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        values = safe_convert_tensor(values)
        self.fit(values)
        return self.transform(values)

    @property
    def obs_max(self):
        with torch.no_grad():
            if self.state_max is not None:
                return self.state_max[..., :self.state_dim].cpu().numpy()
            else:
                return None
    
    @property
    def obs_min(self):
        with torch.no_grad():
            if self.state_min is not None:
                return self.state_min[..., :self.state_dim].cpu().numpy()
            else:
                return None

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump((self.mean, self.std, self.state_dim,
                         self.state_min, self.state_max, self.epsilon), f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.mean, self.std, self.state_dim,self.state_min,\
                self.state_max, self.epsilon = pickle.load(f)


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, hidden_layer_num=5, learning_rate=1e-3, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size

        self.nn_ls = nn.ModuleList([EnsembleFC(state_size + action_size if layer_i==0 else hidden_size, hidden_size, ensemble_size, weight_decay=max(0.000025*layer_i, 0.000075)) for layer_i in range(self.hidden_layer_num-1)])
        # Add variance output
        self.nn_ls.append(EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001))

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        for layer_i, hid_layer in enumerate(self.nn_ls):
            if layer_i == 0:
                nn_output = self.swish(hid_layer(x))
            elif layer_i != len(self.nn_ls)-1:
                nn_output = self.swish(hid_layer(nn_output))
            else:
                nn_output = hid_layer(nn_output)

        mean = nn_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.size()) == len(logvar.size()) == len(labels.size()) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    def __init__(self, network_size, state_size, action_size, reward_size=0, hidden_size=256, hidden_layer_num=3, learning_rate=1e-3, use_decay=False):
        self.network_size = network_size
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, hidden_layer_num, learning_rate=learning_rate, use_decay=use_decay)
        self.ensemble_model.to(device)
        self.scaler = StandardScaler(state_dim=state_size)
    
    def __call__(self, inputs, ret_log_var=False, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = self.ensemble_model(inputs[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=ret_log_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            mean = torch.mean(ensemble_mean, dim=0)
            if ret_log_var:
                ensemble_var = torch.exp(ensemble_var)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            if ret_log_var:
                var = torch.log(var)
            return mean, var
    
    def _collect_loss(self, inputs, labels):
        mean, logvar = self(inputs, ret_log_var=True)
        labels = labels[None, :, :].repeat([self.network_size, 1, 1])
        total_loss, mse_loss = self.ensemble_model.loss(mean, logvar, labels)
        return total_loss, mse_loss

    def train(self, inputs, labels):
        total_loss, mse_loss = self._collect_loss(inputs, labels)
        self.ensemble_model.train(total_loss)
    
    def eval(self, inputs, labels):
        with torch.no_grad():
            total_loss, mse_loss = self._collect_loss(inputs, labels)
        return mse_loss.mean().item()


class FKMInterface:
    def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layer_num=3, network_num=5, lr=1e-3):
        self.predictor = EnsembleDynamicsModel(state_size=state_dim, action_size=action_dim, reward_size=0, hidden_size=hidden_size, hidden_layer_num=hidden_layer_num, network_size=network_num, learning_rate=lr, use_decay=False)
        self._trained = False
        self.model_loss = []
    
    @property
    def trained(self):
        return self._trained
    
    def __call__(self, obs, actions, batch_size=256, deterministic=False, rand_c=True):
        obs = safe_convert_tensor(obs)
        actions = safe_convert_tensor(actions)
        inputs = torch.cat([obs, actions], -1)

        ensemble_model_means, ensemble_model_vars = None, None
        for i in range(0, inputs.size(0), batch_size):
            input = inputs[i:min(i + batch_size, inputs.size(0))]
            b_mean, b_var = self.predictor(input, ret_log_var=False)
            if ensemble_model_means is None:
                ensemble_model_means = b_mean
            else:
                ensemble_model_means = torch.hstack([ensemble_model_means, b_mean])
            if ensemble_model_vars is None:
                ensemble_model_vars = b_var
            else:
                ensemble_model_vars = torch.hstack([ensemble_model_vars, b_var])

        ensemble_model_stds = torch.sqrt(ensemble_model_vars)
        
        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + \
                    torch.rand(size=ensemble_model_means.size()) * ensemble_model_stds       
        
        num_models, batch_size, _ = ensemble_model_means.size()
        model_inds = np.random.choice(range(num_models), size=batch_size)
        batch_inds = np.arange(0, batch_size)
        if rand_c:
            next_obs = ensemble_samples[model_inds, batch_inds]
        else:
            next_obs = ensemble_samples[0, batch_inds]

        return next_obs.squeeze()

    def get_next_state(self, obs, actions, batch_size=256, deterministic=False, rand_c=True):
        with torch.no_grad():
            _state_delta = self(obs, actions, batch_size=batch_size, deterministic=deterministic, rand_c=rand_c)
            if torch.is_tensor(_state_delta):
                _state_delta = _state_delta.cpu().numpy()
            return _state_delta
    
    def train(self, manage_replay_buffer, batch_size, epoch_num=None, max_epochs=50):
        _actions_seq = np.array(manage_replay_buffer.storage[9])
        _actions = np.array(_actions_seq)[:, 0]

        self.predictor.scaler.fit(np.hstack([manage_replay_buffer.storage[0], _actions]))

        # Sample replay buffer
        MAX_SAMPLE_FOR_TRAIN = 50000
        if len(manage_replay_buffer) <= MAX_SAMPLE_FOR_TRAIN:
            states = np.array(manage_replay_buffer.storage[0])
            next_states = np.array(manage_replay_buffer.storage[1])
            actions_seq = np.array(manage_replay_buffer.storage[9])
        else:
            latest_states, latest_next_states, latest_actions_seq = [manage_replay_buffer.storage[_i][-10000:] for _i in [0, 1, 9]]
            states, next_states, _, _, _, _, _, _, _, actions_seq, _ = manage_replay_buffer.sample(MAX_SAMPLE_FOR_TRAIN)
            states = np.vstack([latest_states, states])
            next_states = np.vstack([latest_next_states, next_states])
            actions_seq = np.vstack([latest_actions_seq, actions_seq])

        actions = np.array(actions_seq)[:, 0]
        
        outputs = next_states - states
        inputs = np.concatenate([states, actions], axis=1).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, random_state=0, test_size=0.2, shuffle=True)

        if epoch_num:
            epoch_iter = range(epoch_num)
        else:
            epoch_iter = itertools.count()

        _losses = list()
        _stop_counter = 0
        for epoch in epoch_iter:
            for batch_i in range(0, X_train.shape[0], batch_size):
                train_x, train_y = X_train[batch_i:min(batch_i + batch_size, X_train.shape[0])], y_train[batch_i:min(batch_i + batch_size, X_train.shape[0])]
                train_x, train_y = get_tensor(train_x), get_tensor(train_y)
                self.predictor.train(train_x, train_y)

            if epoch % 5 == 0:
                _val_losses = []
                for batch_i in range(0, X_test.shape[0], batch_size):
                    val_x, val_y = X_test[batch_i:min(batch_i + batch_size, X_test.shape[0])], y_test[batch_i:min(batch_i + batch_size, X_test.shape[0])]
                    val_x, val_y = get_tensor(val_x), get_tensor(val_y)
                    _val_loss = self.predictor.eval(val_x, val_y)
                    _val_losses.append(_val_loss)
                val_loss = np.mean(_val_losses)
                # early stopping
                if len(_losses) > 0 and val_loss >= _losses[-1]:
                    # warning+1
                    _stop_counter += 1
                else:
                    # reset
                    _stop_counter = 0
                _losses.append(val_loss)

                if _stop_counter > 3:
                    break

            if _losses[-1] < 0.01 or epoch > max_epochs:
                break

        self._trained = True
        print('>>>>>>>FKM is trained for {} epochs'.format(epoch))
        return np.mean(_losses)

    def eval(self, manage_replay_buffer, batch_size):
        # Sample replay buffer
        latest_states, latest_next_states, latest_actions_seq = [manage_replay_buffer.storage[_i][-5000:] for _i in [0, 1, 9]]
        states = np.array(latest_states)
        next_states = np.array(latest_next_states)
        actions = np.array([item[0] for item in latest_actions_seq])
        
        outputs = next_states - states
        inputs = np.concatenate([states, actions], axis=1).astype(np.float32)

        _val_losses = []
        for batch_i in range(0, inputs.shape[0], batch_size):
            val_x, val_y = inputs[batch_i:min(batch_i + batch_size, inputs.shape[0])], outputs[batch_i:min(batch_i + batch_size, outputs.shape[0])]
            val_x, val_y = get_tensor(val_x), get_tensor(val_y)
            _val_loss = self.predictor.eval(val_x, val_y)
            _val_losses.append(_val_loss)
        val_loss = np.mean(_val_losses)

        return val_loss
    
    @property
    def scaler(self):
        return self.predictor.scaler

    def save(self, dir, env_name, algo, version, seed):
        torch.save(self.predictor.ensemble_model.state_dict(), "{}/{}_{}_{}_{}_EnsembleDynamicsModel.pth".format(dir, env_name, algo, version, seed))
        self.predictor.scaler.save("{}/{}_{}_{}_{}_EnsembleDynamicsModel_scaler.pkl".format(dir, env_name, algo, version, seed))

    def load(self, dir, env_name, algo, version, seed):
        self.predictor.ensemble_model.load_state_dict(torch.load("{}/{}_{}_{}_{}_EnsembleDynamicsModel.pth".format(dir, env_name, algo, version, seed)))
        self._trained = True
        self.predictor.scaler.load("{}/{}_{}_{}_{}_EnsembleDynamicsModel_scaler.pkl".format(dir, env_name, algo, version, seed))