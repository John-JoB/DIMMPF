import argparse
from Net import PF, RLPF, Redefined_RLPF, Markov_Switching, Polya_Switching, NN_Switching, Erlang_Switching, LSTM, Transformer, DIMMPF,  DIMMPF_redefined, IMMPF
from dpf_rs.model import Simulated_Object, State_Space_Dataset
from trainingRS import test, e2e_train, train_s2s
from dpf_rs.simulation import Differentiable_Particle_Filter
from simulationRS import IMM_Particle_Filter
from dpf_rs.resampling import Soft_Resampler_Systematic, OT_Resampler
from dpf_rs.loss import Supervised_L2_Loss, Magnitude_Loss
from dpf_rs.results import Log_Likelihood_Factors
import torch as pt
from dpf_rs.utils import aggregate_runs, fix_rng
import pickle
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--device', dest='device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device to use')
    parser.add_argument('--alg', dest='alg', type=str, default='DIMMPF', choices=['RLPF', 'LSTM', 'Transformer', 'DIMMPF', 'DIMMPF-OT', 'DIMMPF-N', 'IMMPF'], help='algorithm to use')
    parser.add_argument('--experiment', dest='experiment', type=str, default='Markov', choices=['Markov', 'Polya', 'Exchange', 'Erlang'], help='Experiment to run')
    parser.add_argument('--lr', dest='lr', type=float, default=0.05, help='Initial max learning rate')
    parser.add_argument('--w_decay', dest='w_decay', type=float, default=0.05, help='Weight decay strength')
    parser.add_argument('--lr_steps', dest='lr_steps', nargs='+', type=int, default=[10, 20, 30, 40], help='steps to decrease the lr')
    parser.add_argument('--lr_gamma', dest='lr_gamma', type=float, default=0.5, help='learning rate decay per step')
    parser.add_argument('--clip', dest='clip', type=float, default=10,  help='Value to clip the gradient at')
    parser.add_argument('--lamb', dest='lamb', type=float, default=0.02, help='Ratio of ELBO to MSE loss')
    parser.add_argument('--store_loc', dest='store_loc', type=str, default='temp', help='File in the results folder to store the results dictionary')
    parser.add_argument('--n_runs', dest='n_runs', type=int, default=20, help='Number of runs to average')
    parser.add_argument('--layers', dest='layers', type=int, default=3, help='Number of fully connected layers in neural networks')
    parser.add_argument('--hid_size', dest='hidden_size', type=int, default=11, help='Number of nodes in hidden layers')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='temp', help='Data directory')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Number of epochs to train for')
    
    args = parser.parse_args()
    def create_data():
        nonlocal args
        if args.experiment == 'Markov':
            switching = Markov_Switching(8, 0.8, 0.15, 'Boot', device=args.device)
        elif args.experiment == 'Polya':
            switching = Polya_Switching(8, 'Boot', args.device)
        else: 
            switching = Erlang_Switching(8, 'Boot', args.device)
        model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, switching, 'Boot', args.device)
        sim_obj = Simulated_Object(model, 100, 100, 1, args.device)
        sim_obj.save(f'./data/{args.data_dir}', 50, 20, '', bypass_ask=True)


    if args.alg == 'IMMPF':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)
            if args.experiment == 'Markov':
                model = IMMPF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Markov_Switching(8, 0.8, 0.15, 'Boot', device=args.device), args.device)
            elif args.experiment == 'Polya':
                model = IMMPF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Polya_Switching(8, 'Boot', device=args.device), args.device)
            else:
                model = IMMPF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Erlang_Switching(8, 'Boot', device=args.device), args.device)
            DPF = IMM_Particle_Filter(model, 2000, Soft_Resampler_Systematic(1, 0), 2001, args.device, 'normal')
            loss = Supervised_L2_Loss(function=lambda x : x[:, :, 0:1])
            return test(DPF, loss, 50, data, -1, 0.25)

    if args.alg == 'RLPF':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)

            model = RLPF(8, NN_Switching(8, 8, 'Uni', args.device, 0), 1, args.layers, args.hidden_size, 'Uni', args.device)
            re_model = Redefined_RLPF(model)
            
            
            DPF = Differentiable_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, 0.6), 200, args.device)
            opt = pt.optim.AdamW(params=DPF.parameters(), lr = args.lr, weight_decay=args.w_decay)

            if len(args.lr_steps) > 0:
                opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, args.lr_steps, args.lr_gamma)
            else:
                opt_sch = None
            DPF_ELBO = Differentiable_Particle_Filter(re_model, 200, Soft_Resampler_Systematic(1, 0.6), 200, args.device)
            return e2e_train(DPF, DPF_ELBO, opt, Supervised_L2_Loss(function=lambda x : x[:, :, 0].unsqueeze(2)), 50, data, None, [100, -1, -1], [0.5, 0.25, 0.25], args.epochs, 10, opt_sch, True, args.clip, args.lamb)
        

    if args.alg == 'DIMMPF' or args.alg == 'DIMMPF-OT' or args.alg == 'DIMMPF-N':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)
            model = DIMMPF(8, NN_Switching(8, 8, 'Boot', args.device, 0), 1, args.layers, args.hidden_size, 'Boot', args.device)
    
            if args.alg =='DIMM-OT':
                DPF = IMM_Particle_Filter(model, 200, OT_Resampler(1, 0.001, 100, 0.9), 200, args.device, IMMtype='OT')
            else:
                DPF = IMM_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, 0), 200, args.device, IMMtype='new')

            opt = pt.optim.AdamW(params=DPF.parameters(), lr = args.lr, weight_decay=args.w_decay)
  
            if len(args.lr_steps) > 0:
                opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, args.lr_steps, args.lr_gamma)
            else:
                opt_sch = None
         
            re_model = DIMMPF_redefined(model)
            if args.alg =='DIMM-OT':
                DPF_ELBO = IMM_Particle_Filter(re_model, 200, OT_Resampler(1, 0.001, 100, 0.9), 200, args.device, IMMtype='OT')
            else:
                if args.alg == 'DIMMPF-N':
                    DPF_ELBO = IMM_Particle_Filter(re_model, 200, Soft_Resampler_Systematic(1, 0), 200, args.device, IMMtype='normal')
                else:
                    DPF_ELBO = IMM_Particle_Filter(re_model, 200, Soft_Resampler_Systematic(1, 0), 200, args.device, IMMtype='new')
            return e2e_train(DPF, DPF_ELBO, opt, Supervised_L2_Loss(function=lambda x : x[:, :, 0].unsqueeze(2)), 50, data, None, [100, -1, -1], [0.5, 0.25, 0.25], args.epochs, 10, opt_sch, True, args.clip, True, args.lamb)
        

    if args.alg == 'LSTM' or args.alg == 'Transformer':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)
            if args.alg == 'LSTM':
                NN = LSTM(1, 20, 1, 1, args.device)
            else:
                NN = Transformer(1, 29, 1, 45, 'cuda', 4)
            opt = pt.optim.AdamW(params=NN.parameters(), lr = args.lr, weight_decay=args.w_decay)

            if len(args.lr_steps) > 0:
                opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, args.lr_steps, args.lr_gamma)
            else:
                opt_sch = None
            return train_s2s(NN, opt, data, [100, -1, -1], [0.5, 0.25, 0.25], args.epochs, opt_sch, True, args.clip)
        
    def run():
        nonlocal create_data
        nonlocal train_test
        create_data()
        return train_test()
    fix_rng(1)    
    dic = aggregate_runs(run, args.n_runs, ['loss', 'per_step_loss', 'train_time', 'test_time'])
    print(dic)
    with open(f'./results/{args.store_loc}.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__== '__main__':
    main()



