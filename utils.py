#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/07/17 12:07:09
@Author  :   Bo 
'''
import numpy as np
import os 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import pandas as pd 
from scipy.optimize import curve_fit


# model_path="exp_data/toy_numpy_exp_diff_hession_4/"
model_path = "exp_data/"

def get_shuffle_percent():
    shuffle_percent_grid = []
    for i in range(11)[1:]:
        value = np.linspace((i-1)/10, i/10, 5)
        shuffle_percent_grid+=list(np.round(value[:-1], 3))
    shuffle_percent_grid += [1.0]
    return np.array(shuffle_percent_grid)


def show_convergence_over_multiple_terms(term_group, accuracy_group):
    shuffle_percent = get_shuffle_percent()
    fig = plt.figure(figsize=(12, 6))
    for i, s_accu in enumerate(accuracy_group):
        ax = fig.add_subplot(2, int(np.ceil(len(accuracy_group) / 2)), i+1)
        color = ['r', 'g', 'b', 'c']
        label = ["sigma^2", "zeta", "sigma", "constant"]
        for j, s_value in enumerate(term_group[i]):
            ax.plot(shuffle_percent, s_value, color=color[j], label=label[j])
        ax.legend(loc='best')

    
def calc_convergence(sigma, zeta, L, epsilon, n=10, tau=100, use_mu=False, mu_value=0.0, show=False):
    """Args:
    sigma: sigma^2 
    zeta: zeta^2 
    L: lipschitz constant 
    """
    
    first_term = sigma / n / epsilon
    second_term = np.sqrt(L)*tau*np.sqrt(zeta) / np.sqrt(epsilon)
    third_term = np.sqrt(L)*np.sqrt(tau)*np.sqrt(sigma) / np.sqrt(epsilon)
    fourth = L * tau * np.log(1/epsilon)
    
    if use_mu == True:
        first_term = first_term / mu_value 
        second_term = second_term / mu_value 
        third_term = third_term / mu_value 
        fourth = fourth / mu_value 
    
    if show:
        shuffle_percent = get_shuffle_percent()
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        color = ['r', 'g', 'b', 'c']
        label = ["sigma^2", "zeta", "sigma", "constant"]
        for i, s_value in enumerate([first_term, second_term, third_term, fourth]):
            ax.plot(shuffle_percent, s_value, color=color[i], label=label[i])
        ax.legend(loc='best')
        
    return first_term, second_term, third_term, fourth 


def get_information_group(zeta, sigma, version):
    path = model_path 
    sub_dir = [v for v in os.listdir(path) if "zeta_%d_" % zeta in v and "b_sigma_%.2f_" % sigma in v and "version_%02d_" % version in v]
    if len(sub_dir) > 1:
        return sub_dir 
    else:
        value = np.load(path+sub_dir[0]+"/information_group.npy", allow_pickle=True)
        stat_group = [[] for _ in range(4)]
        for i in range(4):
            stat_group[i] = np.array([v[i] for v in value])
        return np.max(stat_group[0], axis=-1), np.min(stat_group[1], axis=-1), stat_group[2], stat_group[3]
        # return stat_group[0], np.min(stat_group[1], axis=-1), stat_group[2], stat_group[3]
    
    

def get_iteration(zeta, sigma, target_accuracy = 1e-7, s_version=2, low_hessian="original"):    
    exp_dir = model_path 
    sub_dir = sorted([v for v in os.listdir(exp_dir) if ".npy" not in v and ".txt" not in v and "gamma" not in v])
    sub_dir = sorted([v for v in sub_dir if "low_hession_%s" % low_hessian in v])
    sub_dir = [v for v in sub_dir if "zeta_%d_" % zeta in v]
    sub_dir = [v for v in sub_dir if "_b_sigma_%.2f" % sigma in v]
    sub_dir = [v for v in sub_dir if "version_%02d" % s_version in v]
    second_perf_group, perf_iteration_group = [], []
    for v in sub_dir:
        sub_files = sorted([q for q in os.listdir(exp_dir + "/" + v) if "information" not in q])
        if len(sub_files) == 0:
            continue 
        for mm_iter, mm in enumerate(sub_files):
            l = np.load(exp_dir + "/"+v+"/"+mm, allow_pickle=True)
            if mm_iter == 0:
                error_group = [[] for _ in range(len(l))]
            for j in range(len(l)):
                error_group[j].append(l[j])
        best_error_group_2 = []
        best_lr_group_2 = []
        best_iteration_group = []
        for j, s_error in enumerate(error_group):
            select_index = [np.where(np.array(mmm) <= target_accuracy)[0] for mmm in s_error]
            select_index = [v[0] if len(v) > 1 else 1000 for v in select_index]
            best_iteration_group.append(np.min(select_index))
            best_error_group_2.append(s_error[np.argmin(select_index)])
            best_lr_group_2.append(sub_files[np.argmin(select_index)])        
        second_perf_group.append(best_error_group_2)
        perf_iteration_group.append(best_iteration_group)
    return second_perf_group, perf_iteration_group


def calc_avg_iteration(zeta, b_sigma, accuracy_group, version_group):
    iteration_group = []
    for s_version in version_group:
        _out = [get_iteration(zeta, b_sigma, target_accuracy=accu, s_version=s_version, 
                                     low_hessian="original") for accu in accuracy_group]
        _iter_group = [v[1] for v in _out]
        iteration_group.append(_iter_group)
    return np.mean(iteration_group, axis=0)
    
    
def get_different_type_x(accuracy_group, type_use="sec"):
    if type_use == "div_square_epsilon":
        x_value = np.reshape(1 / np.sqrt(accuracy_group), [-1, 1])
    elif type_use == "log_epsilon":
        x_value = np.reshape(np.log(1 / accuracy_group), [-1, 1])
    elif type_use == "div_square_epsilon_with_log_epsilon":
        x_value = np.concatenate([[1 / np.sqrt(accuracy_group)], 
                                  [np.log(1 / accuracy_group)]], axis=0).T 
    elif type_use == "div_epsilon_with_square_epsilon":
        x_value = np.concatenate([[1 / (10 * accuracy_group)], 
                                  [1 / np.sqrt(accuracy_group)]], axis=0).T
    elif type_use == "div_square":
        x_value = np.reshape(1 / 10 / accuracy_group, [-1, 1])
    elif type_use == "all":
        x_value = np.concatenate([[1 / accuracy_group], 
                                  [1 / np.sqrt(accuracy_group)], 
                                  [np.log(1 / accuracy_group)]], axis=0).T
    return x_value 


def fit_con_linear(zeta, b_sigma, type_use="div_square_epsilon", version_group=[0], special_error=[],show=False):
    accuracy_group = list(np.logspace(-6, -4.3, 80))[1:]
    # accuracy_group = list(np.logspace(-6, -4, 80))[1:]
    shuffle_percent = get_shuffle_percent()
    iteration_group = calc_avg_iteration(zeta, b_sigma, accuracy_group, version_group)
    fit_model_group = []
    fitted_params = []
    x_value = get_different_type_x(np.array(accuracy_group), type_use)
    fit_intercept = True #if type_use == "div_square_epsilon" else False 
    for j in range(len(iteration_group[0][0])):
        y_value = np.reshape([v[0][j] for v in iteration_group], [-1, 1])
        if j == 0:
            fit_model = LinearRegression(fit_intercept=fit_intercept).fit(x_value, y_value)
            param = np.array([fit_model.coef_[0][0], fit_model.intercept_[0]])
            fitted_params.append(param)
        else:
            # print(np.shape(x_value[:, 0]), np.shape(y_value[:, 0]), fitted_params[-1])
            param = curve_fitting(x_value[:, 0], y_value[:, 0], fitted_params[-1], shuffle_percent[j])
            fitted_params.append(param)
    # y_pred_group = [standard_func(x_value[:, 0], v[0], v[1]) for v in fitted_params]
    if len(special_error) == 0:
        y_pred_group = [standard_func(x_value[:, 0], v[0], v[1]) for v in fitted_params]
    else:
        y_pred_group = [standard_func(special_error, v[0], v[1]) for v in fitted_params]
    y_value = np.transpose(np.array(iteration_group)[:, 0], (1,0))
    if show:
        if type_use == "log_epsilon" or type_use == "div_square_epsilon":
            print(np.shape(x_value), np.shape(y_value), np.shape(y_pred_group))
            fig = plt.figure(figsize=(10, 4))
            for j in range(20):
                ax = fig.add_subplot(4, 5, j+1)
                ax.plot(x_value[:, 0], y_value[j], 'r-x')
                ax.plot(x_value[:, 0], y_pred_group[j], color='g')
        else:
            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(x_value[:, 0], x_value[:, 1], y_value[0], color='r')
            ax.plot(x_value[:, 0], x_value[:, 1], y_pred_group[0][:, 0], color='g')    
    return fitted_params, y_pred_group, x_value, y_value


def standard_func(x_input, coef, intercept):
    return x_input * coef + intercept


def curve_fitting(x_input, y_input, prev_param, p):
    init_param = prev_param 
    if p <= 0.5:
        init_param[1] -= 1.0
    else:
        init_param[1] -= 0.1
    upper_bounds = list(prev_param)
    lower_bounds = (-np.Inf, -np.Inf)
    parameterbounds = [lower_bounds, upper_bounds]
    
    fittedParameters, pcov = curve_fit(standard_func, x_input, y_input, 
                                       init_param, bounds = parameterbounds)
    return fittedParameters


def create_coef_table(fit_model_group, information_group, n=10, tau=100, type_use="div_square_epsilon_with_log_epsilon", limit_shuffle=[0, 0.5], 
                      fit_type="param", show=False):
    """Args:
    fit_model_group: a group of models 
    information_group: [hession_matrix, mu, zeta, sigma]
    """
    shuffle_percent = get_shuffle_percent()
    index = [i for i,v in enumerate(shuffle_percent) if v <= limit_shuffle[1] and v >= limit_shuffle[0]]
    index = np.array(index)
    hession, mu, zeta_square, sigma_square = information_group 
    hession = hession #** 2
    value_group = []
    theory_value = []
    value_ratio_group = []
    for i in range(len(zeta_square)):
        _value = [sigma_square[i] / n, np.sqrt(hession[i]) * tau * np.sqrt(zeta_square[i]), np.sqrt(hession[i]) * np.sqrt(tau) * np.sqrt(sigma_square[i]), 
                  hession[i] * tau]
        value_group.append(np.array(_value))
    value_group = np.array(value_group)
    for i in range(len(zeta_square)):
        value_group[i,1] = np.sqrt(hession[i]) * (1-shuffle_percent[i])
    
    value_group = value_group[index]
    ratio_group = value_group[1:] / value_group[:1]
    
    if fit_type == "model":
        coef_group = np.array([v.coef_[0] for v in fit_model_group])[index]
        intercept_group = np.array([v.intercept_[0] for v in fit_model_group])[index]
    else:
        coef_group = np.array(fit_model_group)[index, :-1]
        intercept_group = np.array(fit_model_group)[index, -1]
    coef_ratio = coef_group[1:] / coef_group[:1]
 
    value_ratio_title = ["shuffle", "1-shuffle", "sigma_ratio", "hession_ratio", "zeta_ratio", "hession_zeta_ratio", "hession_sigma_ratio"]
    bi = index[0]
    for i in np.arange(len(zeta_square))[index][:-1]:
        if i < len(zeta_square)-1:
            _value_ratio = np.array([sigma_square[i+1] / sigma_square[bi], np.sqrt(hession[i+1]) / np.sqrt(hession[bi]), 
                                    #  mu[bi] / mu[i+1],                                     
                                     np.sqrt(zeta_square[i+1]) / np.sqrt(zeta_square[bi]), 
                            np.sqrt(hession[i+1]) / np.sqrt(hession[bi]) * np.sqrt(zeta_square[i+1]) / np.sqrt(zeta_square[bi]), 
                            np.sqrt(hession[i+1]) / np.sqrt(hession[bi]) * np.sqrt(sigma_square[i+1]) / np.sqrt(sigma_square[bi]), 
                            ])
            value_ratio_group.append(_value_ratio)

    stat_ratio = np.concatenate([np.expand_dims(shuffle_percent[index][1:], axis=1), 
                                 1.0-np.expand_dims(shuffle_percent[index][1:], axis=1), 
                                 np.array(value_ratio_group)], axis=-1)
    stat_ratio_frame = pd.DataFrame(data=np.round(stat_ratio, 3), columns=value_ratio_title)
    
    if type_use == "div_square_epsilon_with_log_epsilon":
        select_index = [1, -1]
        title_use = ["shuffle", "coef_square", "coef_log", "intercept", "square_coef", "log_coef", "square_ratio", "log_ratio"]
    elif type_use == "div_square_epsilon":
        select_index = [1]
        title_use = ["shuffle", "coef", "intercept", "square_coef", "square_ratio"]
    elif type_use == "log_epsilon":
        select_index = [-1]
        title_use = ["shuffle", "coef", "intercept","log_coef", "log_ratio"]
    elif type_use == "all":
        select_index = [0, 1, 2]
    elif type_use == "div_epsilon_with_square_epsilon":
        select_index = [0, 1]
        title_use = ["shuffle", "coef_square", "coef_square_root", "intercept", "coef_square_ratio", 
                     "coef_square_root_ratio", "square_ratio", "square_root_ratio"]
    elif type_use == "div_square":
        select_index = [0]
        title_use = ["shuffle", "coef_square", "intercept", "coef_square_ratio", "square_ratio"]
        
    stat = np.concatenate([np.expand_dims(shuffle_percent[index][1:], axis=1), 
                           coef_group[1:], 
                           np.expand_dims(intercept_group[1:], axis=1),
                           coef_ratio, ratio_group[:, select_index]], axis=-1)
    
    frame = pd.DataFrame(columns=title_use, data=np.round(stat, 3))
    
    if show:
        if type_use == "div_square_epsilon" or type_use == "div_square":
            label_1 = ["Estimated: " + r'$\frac{A_p}{A_0}$', "Calculated: " + r'$\frac{\sqrt{L_{p}}(1-p)}{\sqrt{L_0}}$']
            # label_1 = ["Estimated: " + r'$\frac{A_p}{A_0}$', "Calculated: " + r'$\frac{L_{p}(1-p)}{L_0}$']

            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            ax.plot(shuffle_percent[index][1:], coef_ratio, 'r-x', label=label_1[0])
            ax.plot(shuffle_percent[index][1:], ratio_group[:, select_index], 'g-.', label=label_1[1])
            ax.legend(loc='best', fontsize=14)
            ax.grid(ls=':')
            ax.set_xlabel("shuffle percentage (p)", fontsize=12)
            ax.set_ylabel("Ratio", fontsize=12)
            ax.set_title(r'$R_p=A_p\frac{1}{\sqrt{\epsilon}}+B_p$', fontsize=12)
            # ax = fig.add_subplot(212)        
            # ax.plot(shuffle_percent[index][1:], coef_group[1:], 'm-.', label="coef")
            # ax.plot(shuffle_percent[index][1:], intercept_group[1:], 'b-.', label="intercept")
            # ax.legend(loc='best')
            
        elif type_use == "div_square_epsilon_with_log_epsilon":
            fig = plt.figure(figsize=(7, 5))
            for j in range(2):
                ax = fig.add_subplot(1, 2, j+1)
                ax.plot(shuffle_percent[index][1:], coef_ratio[:, j], 'r-x', label="fitted")
                ax.plot(shuffle_percent[index][1:], ratio_group[:, select_index][:, j], 'g-.', label="calculate")
                ax.legend(loc='best')

    return frame, stat_ratio_frame, coef_group, value_group, index, coef_ratio, ratio_group 


    
    
    
    
         

    