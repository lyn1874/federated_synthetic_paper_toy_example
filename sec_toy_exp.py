#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   toy_exp.py
@Time    :   2023/03/07 18:32:32
@Author  :   Bo 
'''
import numpy as np
import os 
import sys 
import argparse 


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    parser = argparse.ArgumentParser(description='ToyExp')
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--zeta", type=int, default=10)
    parser.add_argument("--gamma", type=int, default=0)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--num_dim", type=int, default=10)
    parser.add_argument("--seed_use", type=int, default=1024)
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--b_sigma", type=float, default=100.0)
    parser.add_argument("--noise_sigma", type=int, default=0)
    parser.add_argument("--low_hession", type=str, default="original")
    return parser.parse_args()


def create_shuffle_full(A, B, shuffle_percentage=0.1):
    num_worker, num_sample, num_dim = np.shape(B)
    num_shuffle = int(num_sample * shuffle_percentage)    
    shuffled_pool = []
    shuffled_A_pool = []
    real_pool = []
    real_A_pool = []
    for i in range(num_worker):
        b_index = np.random.choice(np.arange(num_sample), num_shuffle, replace=False)
        
        shuffled_pool.append(B[i, b_index])
        shuffled_A_pool.append(A[i, b_index])
        
        real_pool.append(B[i, np.delete(np.arange(num_sample), b_index)])
        real_A_pool.append(A[i, np.delete(np.arange(num_sample), b_index)])
        
    shuffled_pool = np.concatenate(shuffled_pool, axis=0)
    shuffled_A_pool = np.concatenate(shuffled_A_pool, axis=0)
    
    shuffled_index = np.random.choice(np.arange(len(shuffled_pool)), len(shuffled_pool), replace=False)
    shuffled_pool = shuffled_pool[shuffled_index]
    shuffled_A_pool = shuffled_A_pool[shuffled_index]
        
    shuffled_data_per_worker_index = np.split(np.arange(len(shuffled_pool)), num_worker)
    shuffled_data_per_worker = [shuffled_pool[v] for v in shuffled_data_per_worker_index]
    
    shuffled_data_A_per_worker = [shuffled_A_pool[v] for v in shuffled_data_per_worker_index]

    
    combine = [np.concatenate([v, q], axis=0) for v, q in zip(real_pool, shuffled_data_per_worker)]
    combine_A = [np.concatenate([v, q], axis=0) for v, q in zip(real_A_pool, shuffled_data_A_per_worker)]
    # combine_A = A
    return shuffled_data_per_worker, real_pool, [np.array(combine_A), np.array(combine)]
    


def generate_functions_more_samples(num_nodes, num_dim, zeta, num_sample, sigma, diff_hession=True, low_hession=True):
    if diff_hession == True:
        A = [np.eye(num_dim) * (i+1) for i in range(num_nodes)]
        A_repeat = [np.repeat([v], num_sample, axis=0) for v in A]
        B = [np.random.normal(0, np.sqrt(zeta) / (i+1) / np.sqrt(num_dim), size=num_dim) for i in range(0, num_nodes)]
        B_mean = B.copy()
        B = [np.random.normal(v, np.ones([len(v)]) * np.sqrt(sigma) / (i+1) / np.sqrt(num_dim) , [num_sample, num_dim]) for i, v in enumerate(B)]
    else:
        A = [np.eye(num_dim) for i in range(num_nodes)]
        A_repeat = [np.repeat([v], num_sample, axis=0) for v in A]
        B = [np.random.normal(0, np.sqrt(zeta) / np.sqrt(num_dim), size=num_dim) for i in range(0, num_nodes)]
        B_mean = B.copy()
        B = [np.random.normal(v, np.ones([len(v)]) * np.sqrt(sigma) / np.sqrt(num_dim) , [num_sample, num_dim]) for i, v in enumerate(B)]
    return A, np.array(A_repeat), np.array(B), np.array(B_mean)


def consensus_distance_multi_sample(X, A, B): # ||x-x*||^2
    if len(np.shape(A)) == 3:
        x_first = np.linalg.inv(np.einsum("ijk,ikl->jl", A, A))
        x_second = [np.einsum("ijk, ij->k", A, B[:, j]) for j in range(np.shape(B)[1])]
        x_star = x_first.dot(np.mean(x_second, axis=0))
    else:
        num_nodes, num_sample = np.shape(A)[:2]
        x_first_first = []
        x_second = []
        for i in range(num_nodes):
            value = np.mean([A[i,j].T.dot(A[i,j]) for j in range(num_sample)], axis=0)
            x_second.append(np.mean([A[i,j].T.dot(B[i,j]) for j in range(num_sample)], axis=0))
            x_first_first.append(value)
        x_first = np.linalg.inv(np.mean(x_first_first, axis=0))
        x_star = x_first.dot(np.mean(x_second, axis=0))        
    if len(X.shape) == 2:
        num_nodes = X.shape[1]
        dist = [np.linalg.norm(X[:,i] - x_star) ** 2 for i in range(0, num_nodes)]
    else:
        dist = [np.linalg.norm(X - x_star)**2]
    return np.mean(dist), x_star


def calc_variance(grad_group):
    avg_grad = np.mean(grad_group, axis=0)
    var = [np.linalg.norm(v - avg_grad, 2)**2 for v in grad_group]
    return np.mean(var)


def optimize_decentralized(X, A_original, B_original, p, gamma, sigma, num_sample, num_iter=100):
    num_nodes, num_dim = np.shape(A_original)[0], np.shape(A_original)[-1]
    if p == 0:
        B = B_original.copy()
        A = A_original.copy()
    else:
        _, _, [A, B] = create_shuffle_full(A_original, B_original, p)
    information_group = get_information(A, B, num_dim, num_nodes)
    _ = estimate_real_zeta(X, A, B)
    x_iter = np.copy(X)
    errors = [consensus_distance_multi_sample(x_iter, A, B)[0]]
    for i in range(0, num_iter):
        if i == 0:
            x_server = np.mean(x_iter, axis=1)
        x_group = []
        for j in range(num_nodes):
            _x = x_server.copy()
            shuffle_index = np.random.choice(np.arange(num_sample), num_sample, replace=False)
            B_shuffle = B[j][shuffle_index]
            if len(np.shape(A)) > 3:
                A_shuffle = A[j][shuffle_index]
            else:
                A_shuffle = A[j]
            for s_iter in range(num_sample):
                if len(np.shape(A)) == 3:
                    grad = A_shuffle.T.dot(A_shuffle.dot(_x) - B_shuffle[s_iter])
                else:
                    grad = A_shuffle[s_iter].T.dot(A_shuffle[s_iter].dot(_x) - B_shuffle[s_iter])
                _x = _x - gamma * grad 
            x_group.append(_x)
        x_server = np.mean(x_group, axis=0)
        errors += [consensus_distance_multi_sample(x_server, A, B)[0]]
    return errors, x_server, information_group
    
    
def estimate_real_zeta(x_init, A, B):
    num_sample = np.shape(B)[1]
    dist, x_star = consensus_distance_multi_sample(x_init, A, B)
    print("The shape of A", np.shape(A), np.shape(B))
    zeta_group = []
    for i in range(np.shape(A)[0]):
        if len(np.shape(A)) == 3:
            first = A[i].dot(x_star) - B[i]
            second = np.mean([A[i].T.dot(first[j]) for j in range(num_sample)], axis=0)
        else:
            second = np.mean([A[i,j].T.dot(A[i,j].dot(x_star) - B[i,j]) for j in range(num_sample)], axis=0)
        ll = np.linalg.norm(second)**2 
        zeta_group.append(ll) 
    print("avg of zetas:", np.mean(zeta_group))
    return x_star, np.mean(zeta_group), zeta_group


def get_grad_at_optimia(A, B, B_center, x_star, print_info=False):
    """Args:
    A: [num_nodes, num_dim, num_dim]
    B: [num_nodes, num_sample, num_dim]
    B_center: b_i, [num_nodes, num_dim], when sample B_{ij}, b_i is the mean 
    x_star: the optimal solution
    """
    num_nodes, num_sample, num_dim = np.shape(B)
    var_over_grad = []
    for i in range(num_nodes):
        if len(np.shape(A)) == 3:
            _grad_over_sample = [A[i].T.dot(A[i].dot(x_star) - B[i,j]) for j in range(num_sample)]
        else:
            _grad_over_sample = [A[i,j].T.dot(A[i,j].dot(x_star) - B[i,j]) for j in range(num_sample)]
        _first = np.mean(_grad_over_sample, axis=0)
        _var_over_grad = np.mean([np.linalg.norm(v - _first, 2)**2 for v in _grad_over_sample], axis=0)
        if len(B_center) > 0:
            _grad_center = A[i].T.dot(A[i].dot(x_star) - B_center[i])
            diff = (_first - _grad_center)
            if print_info:
                print("norm of nabla f_i_x", np.linalg.norm(_first, 2)**2, np.mean([np.linalg.norm(v, 2)**2 for v in _grad_over_sample]))
        var_over_grad.append(_var_over_grad)
    if print_info:
        print("The variance at optimal", ["%.2f" % v for v in var_over_grad], np.mean(var_over_grad))
    return np.mean(var_over_grad)


def estimate_lipschitz(A):
    hession_ei_g = []
    mu_convexity = []
    for i, s_A in enumerate(A):
        s_average = np.mean([s_A[j].T.dot(s_A[j]) for j in range(len(s_A))], axis=0)
        hession_ei = np.max(np.linalg.eigh(s_average)[0])
        mu_convex = np.min(np.linalg.eigh(s_average)[0])
        hession_ei_g.append(hession_ei)
        mu_convexity.append(mu_convex)
    return hession_ei_g, mu_convexity
    
    
def get_information(A_matrix, b_value, num_dim, num_nodes):
    x_init = np.ones([num_dim, num_nodes])
    hession, mu = estimate_lipschitz(A_matrix)
    x_star, zeta, _ = estimate_real_zeta(x_init, A_matrix, b_value)
    
    var = get_grad_at_optimia(A_matrix, b_value, [], x_star, print_info=False)
    return [hession, mu, zeta, var]
    
    
def grid_search(gamma_grid, optimize, target_accuracy, shuffle_percentage):
    positions = []
    all_errors = []
    best_error = np.inf
    backup_index = 0 # if not reached target accuracy - choose the best one reached
    for i, gamma in enumerate(gamma_grid):
        print(gamma)
        errors, _ = optimize(gamma, shuffle_percentage)
        all_errors += [errors]
        errors = np.array(errors)
        errors[np.isnan(errors)] = np.inf
        if np.min(errors) < best_error: # for the backup
            best_error = np.min(errors)
            backup_index = i
        try:
            first_pos = np.nonzero(np.array(errors) < target_accuracy)[0][0]
            positions += [first_pos]
        except:
            positions += [np.inf]
    if np.isinf(np.min(positions)): # target accuracy never reached
        print('\033[93m' + "target accuracy not reached" + '\033[0m')
        print("overall best error achieved:", best_error, "for gamma:", gamma_grid[backup_index])
        return all_errors[backup_index], gamma_grid[backup_index]
    best_index = np.argmin(positions)
    print("best num of iterations:", np.min(positions), "for gamma:", gamma_grid[best_index])
    print("overall best error achieved:", best_error)
    return all_errors[best_index], gamma_grid[best_index]


def run_parallel(num_dim, num_nodes, zeta, version, seed_use, gamma, b_sigma, num_sample, noise_sigma, low_hession):
    model_dir = "exp_data/"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    sub_dir = model_dir + "/nodes_%d_zeta_%d_dim_%d_seed_%d_b_sigma_%.2f_num_sample_%d_sigma_%d_version_%02d_low_hession_%s/" % (
        num_nodes, zeta, num_dim, seed_use, b_sigma, num_sample, noise_sigma, version, low_hession)
    if gamma == 0:
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    gamma_grid = np.logspace(-6, 2, num=40)[:-9]
    print(gamma_grid)
    gamma = gamma_grid[gamma]
    np.random.seed(seed_use)
    A, A_repeat, B, B_center = generate_functions_more_samples(num_nodes, num_dim, zeta, num_sample, b_sigma, diff_hession=True, low_hession=low_hession)
    x_init = np.ones([num_dim, num_nodes])            
    optimizer = lambda gamma, p: optimize_decentralized(x_init, A_repeat, B, p, gamma, 0, num_sample, 800)
    shuffle_percent_grid = []
    for i in range(11)[1:]:
        value = np.linspace((i-1)/10, i/10, 5)
        shuffle_percent_grid+=list(np.round(value[:-1], 3))
    shuffle_percent_grid += [1.0]
    s_error_group = []
    information_group = []
    for i, s_shuffle in enumerate(shuffle_percent_grid):
        error, _, _info = optimizer(gamma, s_shuffle)
        s_error_group.append(error)
        information_group.append(_info)
    np.save(sub_dir + "/gamma_%.6f" % gamma, s_error_group)
    if np.round(gamma, 4) == 0.0127:
        if not os.path.isfile(sub_dir + "/information_group.npy"):
            np.save(sub_dir + "/information_group", information_group)

    

if __name__ == "__main__":
    conf = give_args()
    run_parallel(conf.num_dim, conf.num_nodes, conf.zeta, conf.version, 
                 conf.seed_use, conf.gamma, conf.b_sigma, conf.num_sample, 
                 conf.noise_sigma, conf.low_hession)
