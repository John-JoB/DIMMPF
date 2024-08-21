from typing import Any, Tuple
import torch as pt
from abc import ABCMeta, abstractmethod
from torch.masked import masked_tensor

class Resampler(pt.nn.Module):
    """
    Metaclass for wrapper classes for resampling functions
    To avoid trouble with multiple inheritance this is not explicitly a metaclass, but it should be treated as such
    Constant parameters are most conviently set by their defining at initiation in an overwritten __init__ but this must call super.__init__()
    """

    def forward(self, x_t:pt.Tensor, log_w_t:pt.Tensor):
        """
        Resampling function, to be overwritten in implementation.
        Should take tensors of the particle state and particle weights only.
        And return tensors of the resamapled particles, the new weights and either a tensor of the resampled indicies if it exists or None.
        Care should be taken to propegate gradients correctly as appropriate for the implemented scheme 
        """
        pass

    def to(self, **kwargs):
        if kwargs['device'] is not None:
            self.device = kwargs['device']
        for var in vars(self):
            if isinstance(var, pt.Tensor) and not isinstance(var, pt.nn.Parameter):
                var.to(dtype=kwargs['dtype'], device=kwargs['device'])
        super().to(**kwargs)


def opt_potential(log_a: pt.Tensor, c_potential: pt.Tensor, cost: pt.Tensor, epsilon: float):
    """
        Calculates the update in the Sinkhorn loop for distribution b (either proposal or target)

        Parameters
        -----------
        log_a: (B,N) Tensor
            log of the weights of distribution a 

        c_potential: (B, N) Tensor
            the current potential of distribution a

        cost: (B,N,M) Tensor
            The per unit cost of transporting mass from distribution a to distribution b

        epsilon: float
            Regularising parameter

        Returns
        -----------
        n_potential: (B, M) pt.Tensor
            The updated potential of distribution b

        
    """                 
    temp = log_a.unsqueeze(2) + (c_potential.unsqueeze(2) - cost)/epsilon
    temp = pt.logsumexp(temp, dim=1)
    return -epsilon.squeeze(2)*temp


def sinkhorn_loop(log_a: pt.Tensor, log_b: pt.Tensor, cost: pt.Tensor, epsilon: float, threshold: float, max_iter: int, diam: pt.Tensor, rate: float, device: str='cuda'):
    """
        Calculates the Sinkhorn potentials for entropy regularised optimal transport between two atomic distributions via the Sinkhorn algorithm
        
        Parameters
        ---------------
        log_a: (B,M) Tensor
            log of the weights of the proposal distribution

        log_b: (B,N) Tensor
            log of the weights of the target distribution

        cost: (B,M,N) Tensor
            The per unit cost of transporting mass from the proposal to the target

        epsilon: float
            Regularising parameter

        threshold: float
            The difference in iteratations below which to halt and return

        max_iter: int
            The maximum amount of iterations to run regardless of whether the threshold is hit

        Returns
        ---------------

        f: (B,M) Tensor
            Potential on the proposal

        g: (B,N) Tensor
            Potential on the target

        Notes
        -----------
        Due to convergening to a point, this implementation only retains the gradient at the last step
    """
    i = 1
    f_i = pt.zeros_like(log_a, device=device)
    g_i = pt.zeros_like(log_b, device=device)
    cost_T = pt.einsum('bij -> bji', cost)
    epsilon_ = (pt.ones(f_i.size(0), device=device) * epsilon).reshape(-1, 1, 1)
    epsilon_now = pt.maximum(diam**2, epsilon_)
    continue_criterion = pt.ones(f_i.size(0), device=device, dtype=bool).unsqueeze(1)
    def stop_criterion(i, continue_criterion_):
        return (i < max_iter and pt.any(continue_criterion_))
    with pt.inference_mode():
        while stop_criterion(i, continue_criterion):
            f_u = pt.where(continue_criterion, (f_i + opt_potential(log_b, g_i, cost_T, epsilon_now))/2, f_i)
            g_u = pt.where(continue_criterion, (g_i + opt_potential(log_a, f_i, cost, epsilon_now))/2, g_i)
            update_size = pt.maximum(pt.abs(f_u - f_i), pt.abs(g_u - g_i))
            update_size = pt.max(update_size, dim=1)[0]
            continue_criterion = pt.logical_or(update_size > threshold, epsilon_now.squeeze() > epsilon).unsqueeze(1)
            epsilon_now = pt.maximum(rate*epsilon_now, epsilon_)
            f_i = f_u
            g_i = g_u
            i+=1
    f_i = f_i.clone().detach()
    g_i = g_i.clone().detach()
    epsilon_now = epsilon_now.clone()
    f =  opt_potential(log_b, g_i, cost_T, epsilon_now)
    g = opt_potential(log_a, f_i, cost, epsilon_now)
    return f, g, epsilon_now

def diameter(x: pt.Tensor):
    diameter_x, _ = pt.max(x.std(dim=1, unbiased=False), dim=-1, keepdim=True)
    return pt.where(diameter_x == 0., 1., diameter_x).detach()

def get_transport_from_potentials(log_a: pt.Tensor, log_b: pt.Tensor, cost: pt.Tensor, f: pt.Tensor, g:pt.Tensor, epsilon: float):
    """
    Calculates the transport matrix from the Sinkhorn potentials

    Parameters
    ------------
    
    log_a: (B,M) Tensor
            log of the weights of the proposal distribution

    log_b: (B,N) Tensor
        log of the weights of the target distribution

    cost: (B,M,N) Tensor
        The per unit cost of transporting mass from the proposal to the target

    f: (B,M) pt.Tensor
            Potential on the proposal

    g: (B,N) pt.Tensor
        Potential on the target

    epsilon: float
        Regularising parameter

    Returns
    ---------

    T: (B,M,N) 
        The transport matrix
    """
    log_prefactor = log_b.unsqueeze(1) + log_a.unsqueeze(2)
    #Outer sum of f and g
    f_ = pt.unsqueeze(f, 2)
    g_ = pt.unsqueeze(g, 1)
    exponent = (f_ + g_ - cost)/epsilon
    log_transportation_matrix = log_prefactor + exponent
    return pt.exp(log_transportation_matrix)

def get_sinkhorn_inputs_OT(Nk, log_weights: pt.Tensor, x_t: pt.Tensor, device: str = 'cuda'):
    """
    Get the inputs to the Sinkhorn algorithm as used for OT resampling
    
    Parameters
    -----------
    log_weights: (B,N) Tensor
        The particle weights

    N: int
        Number of particles
    
    x_t: (B,N,D) Tensor
        The particle state

    Returns
    -------------
    log_uniform_weights: (B,N) Tensor
        A tensor of log(1/N)
    
    cost_matrix: (B, N, N) Tensor
        The auto-distance matrix of scaled_x_t under the 2-Norm
    """
    log_uniform_weights = pt.log(pt.ones((log_weights.size(0), Nk), device=device)/Nk)
    centred_x_t = x_t - pt.mean(x_t, dim=1, keepdim=True).detach()
    scale_x = diameter(x_t)
    scaled_x_t = centred_x_t / scale_x.unsqueeze(2)
    cost_matrix = pt.cdist(scaled_x_t, scaled_x_t, 2)**2
    return log_uniform_weights, cost_matrix, scale_x

class transport_grad_wrapper(pt.autograd.Function):
    """
    Wrapper used to clamp the gradient of the transport matrix, for numerical stability
    """
    @staticmethod
    def forward(ctx:Any, x_t:pt.Tensor, log_a:pt.Tensor, transport:pt.Tensor):
        ctx.save_for_backward(x_t, log_a, transport)
        return transport.clone()
    
    @staticmethod
    def backward(ctx, d_dtransport):
        d_dtransport = pt.clamp(d_dtransport, -1., 1.)
        x_t, log_a, transport = ctx.saved_tensors
        d_dx, d_dlog_a = pt.autograd.grad(transport, [x_t, log_a], grad_outputs=d_dtransport, retain_graph=True)
        return d_dx, d_dlog_a, None
    
def apply_transport(x_t: pt.Tensor, transport: pt.Tensor, N:int):
    """
    Apply a transport matrix to a vector of particles

    Parameters
    -------------
    x_t: (B,N,D) Tensor
        Particle state to be transported
    
    transport: (B,M,N) Tensor
        The transport matrix

    N: int
        Number of particles

    """
    return N * pt.einsum('bji, bjd -> bid', transport, x_t)

class OT_Resampler(Resampler):
    """
    OT resampling as described in Corenflos, Thornton et al.
    """

    def __init__(self, epsilon, threshold, max_iter, rate, device:str = 'cuda'):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.threshold = threshold
        self.max_iter = max_iter
        self.rate = rate
    
    def forward(self, Nk:int, x_t:pt.Tensor, log_w_t:pt.Tensor):
        N = x_t.size(1)
        log_b, cost, diam = get_sinkhorn_inputs_OT(N, log_w_t, x_t, self.device)
        diam = x_t.max(dim=1)[0].max(dim=1)[0] - x_t.min(dim=1)[0].min(dim=1)[0]
        f, g, epsilon_used = sinkhorn_loop(log_w_t, log_b, cost, self.epsilon, self.threshold, self.max_iter, diam.reshape(-1, 1, 1), self.rate, self.device)
        if pt.any(pt.isnan(f)) or pt.any(pt.isnan(g)) or pt.any(pt.isnan(epsilon_used)):
            print('loop')
        transport = get_transport_from_potentials(log_w_t, log_b, cost, f, g, epsilon_used)
        transport = transport_grad_wrapper.apply(x_t, log_w_t, transport)
        ratio = N//Nk
        assert N/Nk == ratio
        transport_red = pt.empty((transport.size(0), N, Nk), device=self.device)
        for n in range(Nk):
            transport_red[:, :, n] = pt.sum(transport[:, :,n*ratio:(n+1)*ratio], dim=2)
        #print(pt.sum(transport_red[0], dim=1))
        #print(pt.sum(transport_red[1], dim=0))
        resampled_x_t = apply_transport(x_t, transport_red, Nk)
        
        if pt.any(pt.isnan(resampled_x_t)):
            print(pt.any(pt.isnan(x_t)))
            print(pt.any(pt.isnan(transport)))
            count = 0
            for i in range(transport.size(0)):
                if pt.any(pt.isnan(transport[i])):
                    count += 1
                    print(diam)
                    print(log_w_t[i])
                    print(cost[i])
                    print(i)
            print(count)
            raise SystemExit(0)
            
        return resampled_x_t, pt.zeros((log_w_t.size(0), Nk), device=self.device), None

''' 
class Weird_Resampler(Resampler):
    """
    Modified OT resampling with diagonal chosen greedily
    """

    def __init__(self, epsilon, threshold, max_iter):
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold
        self.max_iter = max_iter
    
    def forward(self, x_t:pt.Tensor, log_w_t:pt.Tensor):
        N = x_t.size(1)
        log_uniform_weights, cost = get_sinkhorn_inputs_OT(log_w_t, N, x_t)
        mask_1 = log_w_t > log_uniform_weights
        mask_2 = pt.logical_not(mask_1)
        truncated_weights = pt.where(mask_1, log_uniform_weights, log_w_t)
        log_a = logsubexp(log_w_t[mask_1], log_uniform_weights[mask_1])
        log_b = logsubexp(log_uniform_weights[mask_2], log_w_t[mask_2])
        transport_mask = pt.einsum('bi,bj->bij', mask_1, mask_2)
        transport = pt.diag_embed(pt.exp(truncated_weights), 0, 1, 2)
        print(transport_mask.size())
        cost = cost[transport_mask]
        print(cost.size())
        f, g = sinkhorn_loop(log_a, log_b, cost, self.epsilon, self.threshold, self.max_iter)
        transport_offdiag = get_transport_from_potentials(log_a, log_b, cost, f, g, self.epsilon)
        transport[transport_mask] = transport_offdiag

        print(transport[0])
        resampled_x_t = apply_transport(x_t, transport, N)
        print(resampled_x_t[0])
        print(pt.sum(transport, dim=1)[0])
        print(pt.sum(transport, dim=2)[0])
        raise SystemExit(1)
        return resampled_x_t, pt.zeros_like(log_w_t), None
''' 
        

def batched_reindex(vector:pt.Tensor, indicies:pt.Tensor):
    '''
    Analagous to vector[indicies], but in a parralelised batched setting.
    '''
    shape = vector.size()
    residual_shape = shape[2:]
    vector_temp = vector.view((shape[0]*shape[1], *residual_shape))
    scaled_indicies = (indicies + pt.arange(shape[0], device=vector.device).unsqueeze(dim=1)*shape[1]).to(pt.int).view(-1)
    vector_temp = vector_temp[scaled_indicies]
    return vector_temp.view((indicies.size(0), indicies.size(1), vector.size(2)))

class soft_grad_wrapper(pt.autograd.Function):
    """
    Wrapper used to clamp the gradient of soft resampling, for numerical stability
    """
    @staticmethod
    def forward(ctx:Any, new_weights:pt.Tensor, new_particles:pt.Tensor, old_weights:pt.Tensor, old_particles:pt.Tensor, grad_scale:pt.Tensor):
        ctx.save_for_backward(new_weights, new_particles, old_weights, old_particles, grad_scale)
        return new_weights.clone(), new_particles.clone()
    
    @staticmethod
    def backward(ctx, d_dweights, d_dxn):
        new_weights, new_particles, old_weights, old_particles, grad_scale = ctx.saved_tensors
        d_dweights = d_dweights*grad_scale
        d_dxn = d_dxn*grad_scale
        d_dw, d_dx = pt.autograd.grad([new_weights, new_particles], [old_weights, old_particles], grad_outputs=[d_dweights, d_dxn], retain_graph=True)
        return None, None, d_dw, d_dx, None
    
class hard_grad_wrapper(pt.autograd.Function):

    @staticmethod
    def forward(ctx:Any, new_particles:pt.Tensor, old_particles:pt.Tensor):
        ctx.save_for_backward(new_particles, old_particles)
        return new_particles.clone()
    
    @staticmethod
    def backward(ctx, d_dxn):
        d_dxn = d_dxn*0.3
        new_particles, old_particles = ctx.saved_tensors
        d_dx = pt.autograd.grad(new_particles, old_particles, grad_outputs=d_dxn, retain_graph=True)[0]
        return None, d_dx

class scale_grad(pt.autograd.Function):
    @staticmethod
    def forward(ctx:Any, input:pt.Tensor, grad_scale):
        ctx.save_for_backward(grad_scale)
        return input.clone()
    
    @staticmethod
    def backward(ctx, d_dinput):
        grad_scale = ctx.saved_tensors[0]
        return grad_scale*d_dinput, None



class Soft_Resampler_Systematic(Resampler):
    """
    Soft resampling with systematic resampler
    """

    def __init__(self, tradeoff:float, grad_scale:float, device:str = 'cuda'):
        super().__init__()
        self.device = device
        self.tradeoff = tradeoff
        self.log_tradeoff = pt.log(pt.tensor((self.tradeoff), device=self.device))
        self.grad_scale = pt.tensor(grad_scale)
        if self.tradeoff != 1:
            self.log_inv_tradeoff = pt.log(pt.tensor((1-self.tradeoff), device=self.device))

    def forward(self, Nk, x_t:pt.Tensor, log_w_t:pt.Tensor):
        B, N, _ = x_t.size()
        
        log_n = pt.log(pt.tensor((N), device=self.device))

        if self.tradeoff == 1:
            log_particle_probs = log_w_t
        else:
            log_particle_probs = pt.logaddexp(self.log_tradeoff+log_w_t,  self.log_inv_tradeoff - log_n)
        
        offset = pt.rand((B), device=self.device)
        cum_probs = pt.cumsum(pt.exp(log_particle_probs.detach()), dim=1)
        cum_probs = pt.where(cum_probs> 1., 1., cum_probs)
        cum_probs[:, -1] = pt.ones((1, B))
        resampling_points = pt.arange(Nk, device=self.device).unsqueeze(dim=0) + offset.unsqueeze(dim=1)
        resampled_indicies = pt.searchsorted(cum_probs*Nk, resampling_points)
        resampled_particles = batched_reindex(x_t, resampled_indicies)
        if self.tradeoff == 1.:
            new_weights = batched_reindex(log_w_t.unsqueeze(2), resampled_indicies).squeeze()
            resampled_particles = resampled_particles
            new_weights = new_weights - new_weights.detach() - log_n
            new_weights = scale_grad.apply(new_weights, self.grad_scale)
        else:
            resampled_particle_probs = batched_reindex(log_particle_probs.unsqueeze(2), resampled_indicies).squeeze()
            resampled_weights = - log_n
            new_weights = resampled_weights - resampled_particle_probs
            new_weights, resampled_particles = soft_grad_wrapper.apply(new_weights, resampled_particles, log_w_t, x_t, self.grad_scale)
        return resampled_particles, new_weights, resampled_indicies
    
    
class Soft_Resampler_Multinomial(Resampler):
    """
    Soft resampling with multinomial resampler
    """

    def __init__(self, tradeoff:float, device:str = 'cuda'):
        super().__init__()
        self.device = device
        self.tradeoff = tradeoff
        if tradeoff != 1:
            self.log_tradeoff = pt.log(pt.tensor((self.tradeoff), device=device))
            self.log_inv_tradeoff = pt.log(pt.tensor((1-self.tradeoff), device=device))

    def forward(self, x_t:pt.Tensor, log_w_t:pt.Tensor):
        N = x_t.size(1)
        
        if self.tradeoff == 1:
            log_particle_probs = log_w_t
        else:
            log_particle_probs = pt.logaddexp(self.log_tradeoff+log_w_t,  self.log_inv_tradeoff - pt.log(pt.tensor((N), device=self.device)))
        particle_probs = pt.exp(log_particle_probs.detach())
        resampled_indicies = pt.multinomial(particle_probs, N, replacement=True).to(device=self.device)
        resampled_particles = batched_reindex(x_t, resampled_indicies)
        if self.tradeoff == 1.:
            new_weights = pt.zeros_like(log_w_t, device=self.device)
        else:
            new_weights = log_w_t - log_particle_probs
            new_weights = batched_reindex(new_weights.unsqueeze(2), resampled_indicies).squeeze()
        return resampled_particles, new_weights, resampled_indicies