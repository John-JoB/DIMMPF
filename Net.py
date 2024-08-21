from typing import Callable
import torch as pt
from dpf_rs.model import *
from numpy import sqrt
from dpf_rs.utils import nd_select, normalise_log_quantity, batched_select         

class Markov_Switching(pt.nn.Module):
    def __init__(self, n_models:int, switching_diag: float, switching_diag_1: float, dyn = 'Boot', device:str ='cuda'):
        super().__init__()
        self.device=device
        self.dyn = dyn
        self.n_models = n_models
        tprobs = pt.ones(n_models) * ((1 - switching_diag - switching_diag_1)/(n_models - 2))
        tprobs[0] = switching_diag
        tprobs[1] = switching_diag_1
        self.switching_vec = pt.log(tprobs).to(device=device)
        self.dyn = dyn
        

    def init_state(self, batches, n_samples):
        if self.dyn == 'Uni':
            self.probs = pt.ones(self.n_models)/self.n_models
        else:
            self.probs = pt.exp(self.switching_vec)
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=self.device).tile((batches, n_samples//self.n_models)).unsqueeze(2)
        return pt.multinomial(pt.ones(self.n_models), batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)

    def forward(self, x_t_1, t):
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=self.device).tile((x_t_1.size(0), x_t_1.size(1)//self.n_models)).unsqueeze(2) 
        shifts = pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0),x_t_1.size(1)])
        new_models = pt.remainder(shifts + x_t_1[:, :, 0], self.n_models)
        return new_models.unsqueeze(2)
    
    def get_log_probs(self, x_t, x_t_1):
        shifts = (x_t[:,:,0] - x_t_1[:,:,0])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return self.switching_vec[shifts]
    
    def get_regime_probs(self, x_t_1):
        ks = pt.arange(0, self.n_models, device=self.device)

        shifts = (ks[None, None, :] - x_t_1[:,:,0:1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return self.switching_vec[shifts].reshape(shifts.size())
    
    def R_0(self, batches, n_samples, k):
        return pt.ones((batches, n_samples, 1), device=self.device)*k
    
    def R_t(self, r_t_1, k):
        return pt.ones_like(r_t_1) * k



class Polya_Switching(pt.nn.Module):
    def __init__(self, n_models, dyn, device:str='cuda') -> None:
        super().__init__()
        self.device = device
        self.dyn = dyn
        self.n_models = n_models
        self.ones_vec = pt.ones(n_models)
        
        
    def init_state(self, batches, n_samples):
        self.scatter_v = pt.zeros((batches, n_samples, self.n_models), device=self.device)
        i_models = pt.multinomial(self.ones_vec, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        return pt.concat((i_models, pt.ones((batches, n_samples, self.n_models), device=self.device)), dim=2)

    def forward(self, x_t_1, t):
        self.scatter_v.zero_()
        self.scatter_v.scatter_(2, x_t_1[:,:,0].unsqueeze(2).to(int), 1)
        c = x_t_1[:,:,1:] + self.scatter_v
        if self.dyn == 'Uni':
            return pt.concat((pt.multinomial(self.ones_vec,  x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0),x_t_1.size(1), 1]), c), dim=2)
        return pt.concat((pt.multinomial(c.reshape(-1, self.n_models), 1, True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        probs = x_t[:, :, 1:]
        probs /= pt.sum(probs, dim=2, keepdim=True)
        s_probs = batched_select(probs, x_t_1[:, :, 1].to(int))
        return pt.log(s_probs)
    

    def get_regime_probs(self, x_t_1):
        probs = x_t_1 / pt.sum(x_t_1, dim=2, keepdim=True)
        return pt.log(probs)
    
    def R_0(self, batches, n_samples, k):
        t = pt.ones((batches, n_samples, self.n_models), device=self.device)
        t[:, :, k] = 2
        return t
    
    def R_t(self, r_t_1, k):
        temp = r_t_1
        temp[: ,:, k] = temp[:, :, k] + 1
        return temp


class Erlang_Switching(pt.nn.Module):
    def __init__(self, n_models, dyn, device:str='cuda') -> None:
        super().__init__()
        self.device = device
        self.dyn = dyn
        self.n_models = n_models
        self.ones_vec = pt.ones(n_models)
        self.permute_backward = pt.remainder(pt.arange(self.n_models) + 1, self.n_models)
        self.permute_forward = pt.remainder(pt.arange(self.n_models) - 1, self.n_models)
        
    def init_state(self, batches, n_samples):
        self.scatter_v = pt.zeros((batches, n_samples, self.n_models), device=self.device)
        
        i_models = pt.multinomial(self.ones_vec, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        return pt.concat((i_models, pt.zeros((batches, n_samples, self.n_models+1), device=self.device)), dim=2)

    def forward(self, x_t_1, t):
        tensor_shape = (x_t_1.size(0), x_t_1.size(1)) 
        self.scatter_v.zero_()
        self.scatter_v.scatter_(2, x_t_1[:,:,0].unsqueeze(2).to(int), 1)
        self.true_probs = (pt.ones(self.n_models, device=self.device) * (0.01/self.n_models)).reshape((1, 1, -1))
        output = x_t_1[:, :, 1:].clone()
        mask = self.scatter_v.to(dtype=bool)
        counts = output[:, :, :-1][mask].reshape(tensor_shape).unsqueeze(2)

        stay_probs = self.scatter_v
        change_probs = self.scatter_v[:, :, self.permute_forward] * 0.6 + self.scatter_v[:, :, self.permute_backward] *0.4
        mixes = (pt.rand(tensor_shape, device = self.device) > 0.01).unsqueeze(2)
        draw_probs = (pt.ones(self.n_models, device=self.device)/self.n_models).reshape((1, 1, -1))

        target_counts = output[:, :, -1].unsqueeze(2)
        self.true_probs = self.true_probs + pt.where(counts == target_counts, change_probs*0.2 + stay_probs*0.8, stay_probs)*(1 - 0.01)

        subtract = (pt.rand(tensor_shape, device=self.device) < 0.2).unsqueeze(2)
        fake_output = output.clone()
        fake_output[:, :, -1] = fake_output[:, :, -1] + 1 
        output = pt.where(subtract, fake_output, output)

        fake_output = output.clone()
        fake_output[:, :, -1] = 0
        fake_output[:, :, :-1] = fake_output[:, :, :-1] + self.scatter_v
        
        target_counts = output[:, :, -1].unsqueeze(2)
        output = pt.where(pt.logical_or(counts < target_counts, pt.logical_not(mixes)), fake_output, output)
        draw_probs = pt.where(pt.logical_and(counts < target_counts, mixes), change_probs, draw_probs)
        
        draw_probs = pt.where(pt.logical_and(counts >= target_counts, mixes), stay_probs, draw_probs)
        if self.dyn == 'Uni':
            return pt.concat((pt.multinomial(self.ones_vec,  x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0),x_t_1.size(1), 1]), output), dim=2)
        return pt.concat((pt.multinomial(draw_probs.reshape(-1, self.n_models), 1, True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), output), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        return self.true_probs    

    def get_regime_probs(self, x_t_1):
        output = pt.ones((x_t_1.size(0), x_t_1.size(1), self.n_models), device = self.device) *(0.01/self.n_models)
        scatter_k = pt.zeros_like(output)
        scatter_k.scatter_(2, x_t_1[:, :, -1].unsqueeze(-1).to(int), 1)
        change_probs = scatter_k[:, :, self.permute_forward] * 0.6 + scatter_k[:, :, self.permute_backward] *0.4
        output = pt.where(x_t_1[:, :, -2:-1] == 0, change_probs*0.2 + scatter_k*0.8, scatter_k) *0.99 + output
        return pt.log(output)
    
    def R_0(self, batches, n_samples, k):
        t = pt.zeros((batches, n_samples, self.n_models+6), device=self.device)
        t[:, :, -1] = k
        return t
    
    def R_t(self, r_t_1, k):
        has_changed = r_t_1.clone()
        view_tensor = batched_select(has_changed, r_t_1[:, :, -1]) 
        view_tensor[:, :] = view_tensor + 1
        has_changed[:, :, -2] = r_t_1[:, :, k]
        has_changed[:, :, -1] = k
        from_mixing = pt.rand((r_t_1.size(0), r_t_1.size(1), 1), device=self.device) < 1/(99*self.n_models + 1)
        mixing_cond = pt.logical_or(k != r_t_1[:, :, -1, None], from_mixing)
        output = pt.where(mixing_cond, has_changed, r_t_1)
        
        
        from_decrease = pt.rand((r_t_1.size(0), r_t_1.size(1), 1), device=self.device) > 0.2
        decrease = r_t_1.clone()
        decrease[:, :, -2] = decrease[:, :, -2] - 1
        output = pt.where(pt.logical_or(mixing_cond, pt.logical_or(from_decrease, r_t_1[:, :, -2:-1] == 0)), output, decrease)

        return output

class NN_Switching(pt.nn.Module):

    def __init__(self, n_models, recurrent_length, dyn, device, softness):
        super().__init__()
        self.device = device
        self.r_length = recurrent_length
        self.n_models = n_models
        self.forget = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.self_forget = pt.nn.Sequential(pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Sigmoid())
        self.scale = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.to_reccurrent = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Tanh())
        self.output_layer = pt.nn.Sequential(pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Tanh(), pt.nn.Linear(recurrent_length, n_models))
        self.dyn = dyn
        self.softness = softness
        

    def init_state(self, batches, n_samples):
        self.probs = pt.ones(self.n_models)/self.n_models
        self.true_probs = pt.ones(self.n_models)/self.n_models
        i_models = pt.multinomial(self.probs, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        if self.r_length > 0:
            return pt.concat((i_models, pt.zeros((batches, n_samples, self.r_length), device=self.device)), dim=2)
        else:
            return i_models

    def forward(self, x_t_1, t):
        old_model = x_t_1[:, :, 0].to(int).unsqueeze(2)
        one_hot = pt.zeros((old_model.size(0), old_model.size(1), self.n_models), device=self.device)
        one_hot = pt.scatter(one_hot, 2, old_model, 1)
        old_recurrent = x_t_1[:, :, 1:]
        c = old_recurrent * self.self_forget(old_recurrent)
        c *= self.forget(one_hot)
        c += self.to_reccurrent(one_hot)
        #if self.dyn == 'Boot':
        probs = pt.abs(self.output_layer(c))
        self.true_probs = probs / pt.sum(probs, dim=2, keepdim=True)
        self.correction = self.softness*self.true_probs.detach() + (1-self.softness)/self.n_models
        probs = self.correction
        return pt.concat((pt.multinomial(probs.reshape(-1, self.n_models), 1, True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
        #return pt.concat((pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_weight(self, x_t, x_t_1):
        models = x_t[:,:,0].to(int)
        probs = batched_select(self.true_probs.reshape(-1, self.n_models), models.flatten()).reshape(x_t.size(0), x_t.size(1))
        corrections = batched_select(self.correction.reshape(-1, self.n_models), models.flatten()).reshape(x_t.size(0), x_t.size(1))
        return pt.log(probs/corrections + 1e-7)
    
    def get_regime_probs(self, r_t_1):
        probs = pt.abs(self.output_layer(r_t_1) + 1e-7)
        probs = probs / pt.sum(probs, dim=2, keepdim=True)
        return pt.log(probs)
    
    def R_0(self, batches, n_samples, k):
        return self.R_t(pt.zeros((batches, n_samples, self.r_length), device=self.device), k)
    
    def R_t(self, r_t_1, k):
        self.zero_vec = pt.zeros(self.n_models, device=self.device)
        self.zero_vec[k] = 1    
        c = r_t_1 *self.self_forget(r_t_1)
        c = c * self.forget(self.zero_vec)
        c = c + self.to_reccurrent(self.zero_vec)
        return c
    

class Recurrent_Unit(pt.nn.Module):
    def __init__(self, input, hidden, output, out_layers):
        super().__init__()
        self.tanh = pt.nn.Tanh()
        self.sigmoid = pt.nn.Sigmoid()
        self.forget = pt.nn.Linear(input, hidden)
        self.to_hidden = pt.nn.Linear(input, hidden)
        self.temper = pt.nn.Linear(input, hidden)
        self.out = Simple_NN(input + hidden, hidden, output, out_layers)

    def forward(self, in_vec, hidden_vec):
        a = hidden_vec * self.sigmoid(self.forget(in_vec))
        b = self.sigmoid(self.temper(in_vec)) * self.tanh(self.to_hidden(in_vec))
        hidden_out = a + b
        out = self.out(pt.concat((in_vec, hidden_out), dim=-1))
        return pt.concat((out, hidden_out), dim=-1)

    
class Likelihood_NN(pt.nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.net = pt.nn.Sequential(pt.nn.Linear(input, hidden), pt.nn.Tanh(), pt.nn.Linear(hidden, hidden), pt.nn.Tanh(), pt.nn.Linear(hidden, output))

    def forward(self, in_vec):
        return self.net(in_vec.unsqueeze(1)).squeeze()


class Simple_NN(pt.nn.Module):
    def __init__(self, input, hidden, output, layers):
        super().__init__()
        nn_layers = [pt.nn.Linear(input, hidden), pt.nn.Tanh()]
        for i in range(layers-2):
            nn_layers += [pt.nn.Linear(hidden, hidden), pt.nn.Tanh()]
        nn_layers += [pt.nn.Linear(hidden, output)]
        self.net = pt.nn.Sequential(*tuple(nn_layers))

    def forward(self, in_vec):
        return self.net(in_vec)
    
class PF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, a:list[int], b:list[int], var_s:float, switching_dyn:pt.nn.Module, dyn ='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.tensor(a, device = device)
        self.b = pt.tensor(b, device = device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.var_factor = -1/(2*var_s)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches:int, n_samples: int):
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)
                                      
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device)
        new_models = self.switching_dyn(x_t_1[:, :, 1:], t)
        index = new_models[:,:,0].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = ((scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t[:,:,1:], x_t_1[:,:,1:])

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        locs = (scaling*pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias)

        return self.var_factor * ((self.y[t] - locs)**2)
    
    def observation_generation(self, x_t):
        noise = self.y_dist.sample((x_t.size(0), 1)).to(device=self.device)
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = ((scaling * pt.sqrt(pt.abs(x_t[:, :, 0])) + bias).unsqueeze(2) + noise)
        return new_pos
    
class IMMPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, a:list[int], b:list[int], var_s:float, switching_dyn:pt.nn.Module, device:str = 'cuda'):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.tensor(a, device = device)
        self.b = pt.tensor(b, device = device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        var_s = pt.tensor(var_s)
        self.var_factor = -1/(2*var_s)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.alg = self.PF_Type.Bootstrap
        self.var_factor = -1/(2*var_s + 1e-6)
        self.pre_factor = -(1/2)*(pt.log(var_s + 1e-6) + pt.log(pt.tensor(2*pt.pi)))

    def M_0_proposal(self, k, batches:int, n_samples: int):
        self.zeros = pt.zeros((batches, n_samples, self.n_models), device=self.device, dtype=bool)
        
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        init_r = self.switching_dyn.R_0(batches, n_samples, k)
        return pt.cat((init_locs, init_r), dim = 2)                 
    
    def M_t_proposal(self, k, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device)
        scaling = self.a[k]
        bias = self.b[k]
        new_pos = ((scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise)
        r = self.switching_dyn.R_t(x_t_1[:, :, 1:], k)
        return pt.cat((new_pos, r), dim = 2).detach()
    
    def log_M_t(self, k, x_t, x_t_1, t: int):
        scaling = self.a[k]
        bias = self.b[k]
        locs = (scaling*x_t_1[:, :, 0] + bias)
        return self.var_factor * ((x_t[:, :, 0] - locs)**2) + self.pre_factor
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        prop_density = self.log_M_t(x_t, x_t_1, t)

        return self.switching_dyn.get_weight(x_t[:,:,1:], x_t_1[:,:,1:]) + prop_density - prop_density.detach()

    def log_f_t(self, k, x_t, t: int):
        scaling = self.a[k]
        bias = self.b[k]
        locs = (scaling*pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias)
        return self.var_factor * ((self.y[t] - locs)**2) + self.pre_factor
    
    def get_regime_probs(self, x_t):
        return self.switching_dyn.get_regime_probs(x_t[:, :, 1:])


class RLPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, init_scale = 1, layers=2, hidden_size = 8, dyn='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        self.switching_dyn = switching_dyn
        for p in self.parameters():
            p =  p * init_scale
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.pi_fact = (1/2)* pt.log(pt.tensor(2*pt.pi))
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def set_x_scaling(self, loc, scale):
        self.x_scale = scale
        self.x_loc = loc

    def M_0_proposal(self, batches:int, n_samples: int):
        self.zeros = pt.zeros((batches, n_samples, self.n_models), device=self.device, dtype=bool)
        self.var_factor = -1/(2*(self.sd_o**2) + 1e-6)
        self.pre_factor = -(1/2)*pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        self.var_factor_dyn = -1/(2*(self.sd_d**2) + 1e-6)
        self.pre_factor_dyn = -(1/2)*pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        init_locs = (self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2) - self.x_loc)/self.x_scale
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        self.scatter = pt.scatter(self.zeros, 2, init_regimes.to(int), True)
        return pt.cat((init_locs, init_regimes), dim = 2).detach(   )                 
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device) * self.sd_d
        new_models = self.switching_dyn(x_t_1[:, :, 1:], t)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=self.device)
        index = new_models[:, :, 0:1].to(int)
        self.scatter = pt.scatter(self.zeros, 2, index, True)
        for m in range(self.n_models):
            mask = self.scatter[:, :, m]
            locs[mask] = self.dyn_models[m](x_t_1[:,:,0:1][mask]).squeeze()
        self.locs = locs
        new_pos = (locs.unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2).detach()
    
    def log_M_t(self, x_t, x_t_1, t: int):
        return self.var_factor_dyn * ((x_t[:, :, 0] - self.locs)**2) + self.pre_factor_dyn
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        prop_density = self.log_M_t(x_t, x_t_1, t)

        return self.switching_dyn.get_weight(x_t[:,:,1:], x_t_1[:,:,1:]) + prop_density - prop_density.detach()

    def log_f_t(self, x_t, t: int):
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        for m in range(self.n_models):
            mask = self.scatter[:, :, m]
            locs[mask] = self.obs_models[m](x_t[:,:,0:1][mask]).squeeze()
        return self.var_factor * ((self.y[t] - locs)**2) + self.pre_factor
    

class Redefined_RLPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, parent:RLPF, device:str = 'cuda'):
        super().__init__(device)
        self.n_models = parent.n_models
        self.dyn_models = parent.dyn_models
        self.obs_models = parent.obs_models
        self.switching_dyn = parent.switching_dyn
        self.sd_d = parent.sd_d
        self.sd_o = parent.sd_o
        
        self.pi_fact = (1/2)* pt.log(pt.tensor(2*pt.pi))
        self.alg = self.PF_Type.Bootstrap


    def set_up(self, state, observations):
        var_factor = -1/(2*(self.sd_o**2) + 1e-6)
        pre_factor = -(1/2)*pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        var_factor_dyn = -1/(2*(self.sd_d**2) + 1e-6)
        pre_factor_dyn = -(1/2)*pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        self.dyn_probs_list = [None]*self.n_models
        self.likelihoods = [None]*self.n_models
        for k in range(self.n_models):
            locs_d = self.dyn_models[k](state[:, :-1, :])
            locs_o = self.obs_models[k](state)
            probs_d = var_factor_dyn * ((state[:, 1:, :] - locs_d)**2) + pre_factor_dyn
            likelihood = var_factor* (observations - locs_o)**2 + pre_factor
            likelihood[:, 1:, :] = likelihood[:, 1:, :] + probs_d
            self.likelihoods[k] = likelihood.squeeze()

    def M_0_proposal(self, batches:int, n_samples: int):
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        self.zeros = pt.zeros((batches, n_samples, self.n_models), device=self.device, dtype=bool)
        self.scatter = pt.scatter(self.zeros, 2, init_regimes.to(int), True)
        return init_regimes                
    
    def M_t_proposal(self, x_t_1, t: int):
        new_models = self.switching_dyn(x_t_1, t)
        return new_models
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_weight(x_t[:,:,1:], x_t_1[:,:,1:])

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 0:1].to(int)
        self.scatter = pt.scatter(self.zeros, 2, index, True)
        probs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        for m in range(self.n_models):
            mask = self.scatter[:, :, m]
            probs[mask] = (self.likelihoods[m][:, t, None].expand(-1, mask.size(1)))[mask]
        return probs
    

class DIMMPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, init_scale = 1, layers=2, hidden_size = 8, dyn='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        self.switching_dyn = switching_dyn
        for p in self.parameters():
            p =  p * init_scale
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.pi_fact = (1/2)* pt.log(pt.tensor(2*pt.pi))
        self.alg = self.PF_Type.Bootstrap

    def set_x_scaling(self, loc, scale):
        self.x_scale = scale
        self.x_loc = loc

    def M_0_proposal(self, k, batches:int, n_samples: int):
        self.zeros = pt.zeros((batches, n_samples, self.n_models), device=self.device, dtype=bool)
        self.var_factor = -1/(2*(self.sd_o**2) + 1e-6)
        self.pre_factor = -(1/2)*pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        self.var_factor_dyn = -1/(2*(self.sd_d**2) + 1e-6)
        self.pre_factor_dyn = -(1/2)*pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        init_locs = (self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2) - self.x_loc)/self.x_scale
        init_r = self.switching_dyn.R_0(batches, n_samples, k)
        return pt.cat((init_locs, init_r), dim = 2)             
    
    def M_t_proposal(self, k, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device) * self.sd_d
        locs = self.dyn_models[k](x_t_1[:,:,0:1])
        new_pos = (locs + noise)
        r = self.switching_dyn.R_t(x_t_1[:, :, 1:], k)
        return pt.cat((new_pos, r), dim = 2)
    
    def log_M_t(self, k, x_t, x_t_1, t: int):
        locs = self.dyn_models[k](x_t_1[:,:,0:1]).squeeze()
        locs = locs[:, None, :]
        return self.var_factor_dyn * ((x_t[:, :, None, 0] - locs)**2) + self.pre_factor_dyn
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_f_t(self, k, x_t, t: int):
        locs = self.obs_models[k](x_t[:,:,0:1])
        return (self.var_factor * ((self.y[t][:, None, :] - locs)**2) + self.pre_factor).squeeze()
    
    def get_regime_probs(self, x_t):
        return self.switching_dyn.get_regime_probs(x_t[:, :, 1:])
    
    
class DIMMPF_redefined(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, parent:DIMMPF, device:str = 'cuda'):
        super().__init__(device)
        self.n_models = parent.n_models
        self.dyn_models = parent.dyn_models
        self.obs_models = parent.obs_models
        self.switching_dyn = parent.switching_dyn
        self.sd_d = parent.sd_d
        self.sd_o = parent.sd_o
        
        self.pi_fact = (1/2)* pt.log(pt.tensor(2*pt.pi))
        self.alg = self.PF_Type.Bootstrap

    def set_up(self, state, observations):
        var_factor = -1/(2*(self.sd_o**2) + 1e-6)
        pre_factor = -(1/2)*pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        var_factor_dyn = -1/(2*(self.sd_d**2) + 1e-6)
        pre_factor_dyn = -(1/2)*pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        self.dyn_probs_list = [None]*self.n_models
        self.likelihoods = [None]*self.n_models
        for k in range(self.n_models):
            locs_d = self.dyn_models[k](state[:, :-1, :])
            locs_o = self.obs_models[k](state)
            probs_d = var_factor_dyn * ((state[:, 1:, :] - locs_d)**2) + pre_factor_dyn
            likelihood = var_factor* (observations - locs_o)**2 + pre_factor
            likelihood[:, 1:, :] = likelihood[:, 1:, :] + probs_d
            self.likelihoods[k] = likelihood.squeeze()

    def M_0_proposal(self, k, batches:int, n_samples: int):
        self.zeros = pt.zeros((batches, n_samples, self.n_models), device=self.device, dtype=bool)
        
        init_r = self.switching_dyn.R_0(batches, n_samples, k)
        return init_r         
    
    def M_t_proposal(self, k, x_t_1, t: int):
        r = self.switching_dyn.R_t(x_t_1, k)
        return r
    
    def log_M_t(self, k, x_t, x_t_1, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1), 1), device='cuda')
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_f_t(self, k, x_t, t: int):
        return self.likelihoods[k][:, t:t+1].expand(-1, x_t.size(1))
    
    def get_regime_probs(self, x_t):
        return self.switching_dyn.get_regime_probs(x_t)


class LSTM(pt.nn.Module):

    def __init__(self, obs_dim, hid_dim, state_dim, n_layers, device='cuda') -> None:
        super().__init__()
        self.lstm = pt.nn.LSTM(obs_dim, hid_dim, n_layers, True, True, 0.0, False, state_dim, device)

    def forward(self, y_t):
            return self.lstm(y_t)[0]
        

class Transformer(pt.nn.Module):

    def __init__(self, obs_dim, hid_dim, state_dim, T:int = 50, device ='cuda', layers = 2):
        super().__init__()
        self.encoder_layer = pt.nn.TransformerEncoderLayer(hid_dim, 1, hid_dim, 0.1, batch_first=True, device=device)
        self.transformer = pt.nn.TransformerEncoder(self.encoder_layer, layers)
        self.encoding = pt.nn.Linear(obs_dim, hid_dim, device=device)
        self.decoding = pt.nn.Linear(hid_dim, state_dim, device=device)
        self.relu = pt.nn.ReLU()
        self.mask = pt.tril(pt.ones((T+1, T+1), device=device))
    
    def forward(self, y_t):
        t = self.encoding(y_t)
        t = self.relu(t)
        t = self.transformer(t, mask = self.mask, is_causal = True)
        return self.decoding(t)
