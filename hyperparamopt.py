import hyperopt as hypo
import argparse
from Net import PF, RSDBPF, Redefined_RSDBPF, Markov_Switching, Polya_Switching, Erlang_Switching, NN_Switching, DBPF, LSTM, MAPF, Transformer, DIMMPF, DIMMPF_SVM, DIMMPF_redefined
from dpf_rs.model import Simulated_Object, State_Space_Dataset
from trainingRS import test, e2e_train, e2e_likelihood_train, train_s2s
from dpf_rs.simulation import Differentiable_Particle_Filter
from simulationRS import MADPF, IMM_Particle_Filter
from dpf_rs.resampling import Soft_Resampler_Systematic
from dpf_rs.loss import Supervised_L2_Loss, Magnitude_Loss
from dpf_rs.results import Log_Likelihood_Factors
import torch as pt
from dpf_rs.utils import aggregate_runs, fix_rng
import pickle
import numpy as np

def optimise(function, space, max_evals):
    t = hypo.Trials()
    best = hypo.fmin(fn=function, space=space, algo=hypo.tpe.suggest, max_evals=max_evals, trials=t)
    return best

transformer_space = {'lr': hypo.hp.lognormal('_lr', -3, 1),
         'w_decay': hypo.hp.lognormal('_w_decay', -3, 1),
         'lr_gamma': hypo.hp.uniform('_lr_gamma', 0.5, 1),
         'clip': hypo.hp.loguniform('_clip', 0, 3),
         'hidden_size': hypo.hp.quniform('_hid_size', 4, 30, 1),
         'T': hypo.hp.quniform('_T', 5, 60, 1),
         'layers': hypo.hp.quniform('_layers', 1, 5, 1)}

RLPF_space = {'lr': hypo.hp.lognormal('_lr', -3, 1),
         'w_decay': hypo.hp.lognormal('_w_decay', -3, 1),
         'lr_gamma': hypo.hp.uniform('_lr_gamma', 0.5, 1),
         'clip': hypo.hp.loguniform('_clip', 0, 3),
         'init_scale': hypo.hp.lognormal('_init_scale', 0, 1),
         'lamb': hypo.hp.normal('_lamb', 2, 1.5),
         'soft_choice': hypo.hp.choice('_soft_choice', [{'softness' : 1}, {'softness': hypo.hp.uniform('_softness', 0.3, 1)}]),
         'grad_decay': hypo.hp.uniform('_grad_decay', 0, 1),
         'layers_info': hypo.hp.choice('_layers_info', [{'layers' : 2,  'hid_size': hypo.hp.qnormal('_hid_size_1', 20, 10, 1)}, {'layers' : 3, 'hid_size' : hypo.hp.quniform('_hid_size_2', 4, 15, 1)}])}

IMMPF_space = {#'lr': hypo.hp.lognormal('_lr', -3, 1),
         'w_decay': hypo.hp.lognormal('_w_decay', -3, 1),
         'init_scale': hypo.hp.lognormal('_init_scale', 0, 1),
         #'grad_decay': hypo.hp.uniform('_grad_decay', 0, 1),
         'lambda': hypo.hp.lognormal('_lambda', -4.6, 0.6),
         'dr': hypo.hp.quniform('_dr', 6, 12, 1)}

def runRLPF(param_dict):
    print(param_dict)
    

    try:
        data = State_Space_Dataset(f'./data/hyp_opt', lazy = False, device='cuda', num_workers=0)
        model = RSDBPF(8, NN_Switching(8, 8, 'Uni', 'cuda', 0), param_dict['init_scale'], 3, 11, 'Uni', 'cuda')
        re_model = Redefined_RSDBPF(8, NN_Switching(8, 8, 'Uni', 'cuda', 0), 'Uni', 'cuda')
        DPF = Differentiable_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, param_dict['grad_decay']), 200, 'cuda')
        opt = pt.optim.AdamW(params=DPF.parameters(), lr = param_dict['lr'], weight_decay=param_dict['w_decay'])
        DPF_ELBO = Differentiable_Particle_Filter(re_model, 200, Soft_Resampler_Systematic(1, param_dict['grad_decay']), 200, 'cuda')
        opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, [5, 10, 20, 30, 40], param_dict['lr_gamma'])
        _, loss = e2e_likelihood_train(DPF_ELBO, DPF, opt, 50, data, [100, -1, -1], [0.5, 0.25, 0.25], 20, 10, opt_sch, False, 10, abs(param_dict['lamb']))
    except:
        return 10000
    
    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
        return 10000
    return np.mean(loss)


def runIMMPF(param_dict):
    print(param_dict)
    
    data = State_Space_Dataset(f'./data/hyp_opt', lazy = False, device='cuda', num_workers=0)
    model = DIMMPF(8, NN_Switching(8, int(param_dict['dr']), 'Uni', 'cuda', 1), param_dict['init_scale'], 3, 11, 'Boot', 'cuda')
    DPF = IMM_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, 0), 201, 'cuda', 'new', 1)
    opt = pt.optim.AdamW(params=DPF.parameters(), lr=0.05, weight_decay=param_dict['w_decay'])
    #opt = pt.optim.SGD(params=DPF.parameters(), lr = param_dict['lr'], weight_decay=param_dict['w_decay'])
    re_model = DIMMPF_redefined(model)
    DPF_ELBO = IMM_Particle_Filter(re_model, 200, Soft_Resampler_Systematic(1, 0), 200, 'cuda', IMMtype='new', grad_scale=0.9)

    opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, [5, 10, 20, 30, 40], 0.5)
    loss = Supervised_L2_Loss(function=lambda x : x[:, :, 0].unsqueeze(2))
    _, loss, _, _ = e2e_train(DPF, DPF_ELBO, opt, loss, 50, data, None, [100, -1, -1], [0.5, 0.25, 0.25], 50, 10, opt_sch, False, 10, True, param_dict['lambda'])
    try:
        pass
    except Exception as err:
        print(err)
        return 10000
    
    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
        return 10000
    return np.mean(loss)



def runIMMPF_Exchange(param_dict):
    print(param_dict)
    

    try:
        data_train = State_Space_Dataset(f'./data/exchange/trajectories', prefix='train', lazy = False, device='cuda', num_workers=0)
        data_test = State_Space_Dataset(f'./data/exchange/trajectories', prefix='test', lazy = False, device='cuda', num_workers=0)
        model = DIMMPF(8, NN_Switching(8, param_dict['dr'], 'Uni', 'cuda', 1), param_dict['init_scale'], int(param_dict['layers_info']['layers']), int(abs(param_dict['layers_info']['hid_size'])), 'Boot', 'cuda')
        DPF = IMM_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, 0), 201, 'cuda', 'new', param_dict['grad_decay'])
        opt = pt.optim.AdamW(params=DPF.parameters(), lr = param_dict['lr'], weight_decay=param_dict['w_decay'])
        loss = Magnitude_Loss(Log_Likelihood_Factors(), sign=-1)
        _, loss = e2e_train(DPF, opt, loss, 99, data_train, data_test, [100, -1, -1], [0.5, 0.25, 0.25], 50, 10, None, False, param_dict['clip'], False)
    except Exception as err:
        print(err)
        return 10000
    
    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
        return 10000
    return np.mean(loss)



def run_transformer(param_dict):
    


    data = State_Space_Dataset(f'./data/hyp_opt', lazy = False, device='cuda', num_workers=0)
    NN = Transformer(1, int(param_dict['hidden_size']), 1, int(param_dict['T']), 'cuda', int(param_dict['layers']))
    opt = pt.optim.AdamW(params=NN.parameters(), lr = param_dict['lr'], weight_decay=param_dict['w_decay'])
    opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, [10, 20, 30, 40], param_dict['lr_gamma'])
    _, loss, _, _ = train_s2s(NN, opt, data, [100, -1, -1], [0.5, 0.25, 0.25], 50, opt_sch, False, param_dict['clip'])
    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
        return 10000
    return np.mean(loss)



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--alg', dest='alg', type=str, default='RSPF', choices=['RLPF', 'RSDBPF', 'RSPF', 'DBPF', 'LSTM', 'MADPF', 'Transformer', 'IMMPF'], help='algorithm to use')
    parser.add_argument('--evals', dest='evals', type=int, default=50, help='number of evals')
    parser.add_argument('--experiment', dest='experiment', type=str, default='Markov', choices=['Markov', 'Polya', 'Exchange', 'Erlang'], help='Experiment to run')
    args = parser.parse_args()
    if args.experiment == 'Markov':
        model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Markov_Switching(8, 0.8, 0.15, 'Boot', device='cuda'), 'Boot', 'cuda')
    if args.experiment == 'Polya':
        model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Polya_Switching(8, 'Boot', device='cuda'), 'Boot', 'cuda')
    if args.experiment == 'Erlang':
        model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Erlang_Switching(8, 'Boot', device='cuda'), 'Boot', 'cuda')
    if not args.experiment == 'Exchange':
        sim_obj = Simulated_Object(model, 100, 100, 1, 'cuda')
    #sim_obj.save(f'./data/hyp_opt', 50, 20, '', bypass_ask=True)

    
    if args.alg == 'RLPF':
        fun = runRLPF
        space = RLPF_space
    if args.alg == 'Transformer':
        fun = run_transformer
        space = transformer_space
    if args.alg == 'IMMPF':
        fun = runIMMPF
        space = IMMPF_space
    print(optimise(fun, space, args.evals))



main()
