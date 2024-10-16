#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   make_figure.py
@Time    :   2023/07/17 12:08:34
@Author  :   Bo 
'''
import numpy as np 
import os 
import matplotlib 
import matplotlib.ticker as mticker
FONTSIZE = 7
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.title_fontsize": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        # "text.fontsize": FONTSIZE,
        
    }
)
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import seaborn as sns 
import utils as teu 
def form3(x, pos):
    """ This function returns a string with 3 decimal places, given the input x"""
    return '%.1f' % x


def ax_global_get(fig):
    ax_global = fig.add_subplot(111, frameon=False)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    return ax_global


def get_shuffle_percent():
    shuffle_percent_grid = []
    for i in range(11)[1:]:
        value = np.linspace((i-1)/10, i/10, 5)
        shuffle_percent_grid+=list(np.round(value[:-1], 3))
    shuffle_percent_grid += [1.0]
    return np.array(shuffle_percent_grid)


def get_third_figure(ax):
    #seed_use = 12988 
    version=6
    zeta = 10 
    sigma = 0 
    information_group = teu.get_information_group(zeta, sigma, version)   
    type_use = "div_square_epsilon"
    fit_model_group, y_pred_group, x_value, y_value = teu.fit_con_linear(zeta, sigma, type_use, 
                                                                     [version], show=False)

    _, _, _, _, index, coef_ratio, ratio_group = teu.create_coef_table(fit_model_group, information_group, type_use=type_use,
                                                       limit_shuffle=[0.0, 0.8], fit_type="param", show=False)
    if not ax:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
    shuffle_percent = get_shuffle_percent()
    theory_label = r'$\left(1-p + p\frac{L_\mathrm{avg}}{L_{\mathrm{max}}}\right)(1-p)$'
        
    theory_label1 = r'$(1-p)\sqrt{1-p+p\frac{L_\mathrm{avg}}{L_{\mathrm{max}}}}$'
    theory_label2 = "Theory"
    
    label_1 = ["Estimate: " + r'$\frac{A_p}{A_{p=0}}$', "Theory: " + theory_label]

    select_index = [1]
    ax.plot(shuffle_percent[index][1:], coef_ratio, 'r', label=label_1[0], lw=1)
    ax.plot(shuffle_percent[index][1:], ratio_group[:, select_index], 'g', label=label_1[1], lw=1)
    ax.text(0.0, 0.4, label_1[0], color='r')
    ax.text(0.12, 0.85, theory_label1, color='g')
    ax.text(0.38, 0.7, theory_label2, color='g')
    ax.grid(ls=':')
    ax.set_xlabel("shuffle percentage (p) \n (c)", labelpad=3)
    ax.tick_params(axis='y', which='major', pad=0)
    ax.tick_params(axis='y', which='minor', pad=0)
    ax.set_ylabel("Ratio", labelpad=0)
    ax.set_title(r'$R_p=A_p\frac{1}{\sqrt{\epsilon}}+B_p$', pad=-1)



def create_influence_of_shuffling_plot(ax):
    target_perf = 1.160264386341429e-06
    version = 6
    use_performance, _, _ = get_convergence([10], [0], target_perf, [version], show=False)
    extra_ticks=[166.5757977 , 143.7145909 , 112.72622441,  89.36373112,
         62.43194285]
    extra_ticks = np.array(extra_ticks)[[0, 2, 4]]
    index = [0,  8,  16]    

    if not ax:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
    color = sns.color_palette("flare", 8)[1:-1][::-1]
    color = ['r', 'g', 'b', 'orange', 'm']
    p_value = "p"
    label_group = [p_value+'=0.0',  p_value+'=0.2', p_value+'=0.4']
    
    for j, s_error in enumerate(np.array(use_performance[0])[index]):
        ax.plot(s_error, color=color[j], label=label_group[j], lw=0.4)        
        ax.plot([extra_ticks[j], extra_ticks[j]], [6e-7, 2e-6], color=color[j], lw=1)
    ax.legend(loc='center left', frameon=False, handlelength=0.85, labelspacing=0.02,
                bbox_to_anchor=(0.61, 0.74), ncol=1, handletextpad=0.4)
    ax.plot([1, 1000], [target_perf, target_perf], ls='--', color='k', lw=0.5)
    
    ax.set_yscale("log")    
    ax.set_xscale("log")
    ax.set_xlim((10, 400))
    ax.set_xlabel("Rounds\n (b)", labelpad=0)
    ax.grid(ls=':')
    ax.set_title(r'$\zeta^2$' +"=10" + "; " + r'$\sigma^2$' + "=0", pad=-1)
    return np.array(use_performance[0])[index]
        

def get_convergence(zeta, sigma=[], target_accuracy = 1e-7, version_use=[], low_hession="original", dataset="toy", show=False):
    exp_dir = "../exp_data/toy_numpy_exp_diff_hession_4/"
    sub_dir = sorted([v for v in os.listdir(exp_dir) if ".npy" not in v and ".txt" not in v and "gamma" not in v])
    sub_dir = sorted([v for v in sub_dir if "low_hession_%s" % low_hession in v])
    if len(zeta) > 0:
        sub_dir = [v for v in sub_dir if "zeta_%d_" % zeta[0] in v]
    if len(sigma) > 0:
        sub_dir = [v for v in sub_dir if "_b_sigma_%.2f" % sigma[0] in v]
    if len(version_use) > 0:
        sub_dir = [v for v in sub_dir if "version_%02d" % version_use[0] in v]

    first_perf_group, second_perf_group = [], []
    lr_group_tot = []
    iteration_group_tot = []
    for v in sub_dir:
        sub_files = sorted([q for q in os.listdir(exp_dir + "/" + v) if "information"  not in q])
        if len(sub_files) == 0:
            continue 
        for mm_iter, mm in enumerate(sub_files):
            l = np.load(exp_dir + "/"+v+"/"+mm, allow_pickle=True)
            if mm_iter == 0:
                error_group = [[] for _ in range(len(l))]
            for j in range(len(l)):
                error_group[j].append(l[j])
        best_error_group = []
        best_lr_group = []
        for j, s_error in enumerate(error_group):
            select_index = np.nanargmin([np.nanmean(mmm[-10:]) for mmm in s_error])
            best_error_group.append(s_error[select_index])
            best_lr_group.append(sub_files[select_index])

        best_error_group_2 = []
        best_lr_group_2 = []
        iteration_group = []
        for j, s_error in enumerate(error_group):
            select_index = [np.where(np.array(mmm) <= target_accuracy)[0] for mmm in s_error]
            select_index = [v[0] if len(v) > 1 else 1000 for v in select_index]
            iteration_group.append(np.min(select_index))
            best_error_group_2.append(s_error[np.argmin(select_index)])
            best_lr_group_2.append(sub_files[np.argmin(select_index)])        

        first_perf_group.append(best_error_group)
        second_perf_group.append(best_error_group_2)
        lr_group_tot.append([float(s_gamma.split("gamma_")[1].split(".npy")[0]) for s_gamma in best_lr_group_2])
        iteration_group_tot.append(iteration_group)

    use_perf = second_perf_group    
    return use_perf, iteration_group_tot, lr_group_tot  
