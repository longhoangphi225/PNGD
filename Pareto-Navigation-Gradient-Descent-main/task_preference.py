import numpy as np
import torch

from problems.toy_biobjective import circle_points, concave_fun_eval, create_pf
from solvers import epo_search, pareto_mtl_search, linscalar, moo_mtl_search, NPO_solver

import matplotlib.pyplot as plt
from latex_utils import latexify

# algorithms
# methods = ['EPO']
methods = ['PNG']

# parameters
alpha = 0.25
threshold = 0.1


result_all = {}

K = 4       # Number of trajectories
n = 10      # dim of solution space
m = 2       # dim of objective space
rs = circle_points(K)  # preference
pmtl_K = 5
pmtl_refs = circle_points(pmtl_K, 0, np.pi / 2)

color_list = ['#28B463', '#326CAE', '#FFC300', '#FF5733']
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
                'size': 18,
               }

# fixed parameter
start = -1
step_size = 0.2
max_iters = 150

ss, mi = 0.1, 100

#
for method in methods:
    result_all[method] = []
    last_ls = []
    for k, r in enumerate(rs):
        r_inv = 1. / r
        ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)

        x0 = np.zeros(n)
        x0[range(0, n, 2)] = 0.3
        x0[range(1, n, 2)] = -0.3
        # x0 += 0.1 * np.random.randn(n)
        x0 = np.random.uniform(-0.6, 0.6, n) if method in ["MOOMTL", "LinScalar"] else x0
        if method == 'EPO':
            _, res = epo_search(concave_fun_eval, r=r, x=x0,
                                step_size=ss, max_iters=150)
        if method == 'PMTL':
            _, res = pareto_mtl_search(concave_fun_eval,
                                       ref_vecs=pmtl_refs, r=r_inv, x=x0,
                                       step_size=0.2, max_iters=150)
        if method == 'LinScalar':
            _, res = linscalar(concave_fun_eval, r=r, x=x0,
                               step_size=ss, max_iters=mi)
        if method == 'MOOMTL':
            _, res = moo_mtl_search(concave_fun_eval, x=x0,
                                    step_size=0.2, max_iters=150)

        if method == 'PNG':
            e_solver = NPO_solver(max_iters=max_iters, n_dim=n, step_size=step_size)
            x0 = torch.tensor(list(x0)).float().unsqueeze(0)
            x0.requires_grad = True
            _, res = e_solver.optimize_epo(x0, list(r),  alpha=alpha, threshold=threshold)

        result_all[method].append(res)


latexify(fig_width=5., fig_height=5.)
ss, mi = 0.1, 100
pf = create_pf()
fig, ax = plt.subplots()
# fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.97)

# plot ray
for k, r in enumerate(rs):
    r_inv = 1. / r
    ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
    ep_ray_line = np.stack([np.zeros(m), ep_ray])
    label = r'$r^{-1}$ ray' if k == 0 else ''
    ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color=color_list[k],
            lw=2, ls='--', dashes=(15, 5))


for method in methods:
    last_ls = []

    if method == 'EPO':
        shape = ','
        size = 100
    if method == 'PNG':
        shape = ','
        size = 100

    # plot the trajectory
    for k, r in enumerate(rs):
        res = result_all[method][k]['ls']
        if k == 0:
            ax.scatter(res[-1, 0], res[-1, 1], s=size, c=color_list[0], marker=shape, alpha=1, label=method)
        else:
            ax.scatter(res[-1, 0], res[-1, 1], s=size, c=color_list[0], marker=shape, alpha=1)
        ax.plot(res[:, 0], res[:, 1], c=color_list[k], alpha=1, lw=2)

ax.set_xlabel(r'$l_1$')
ax.set_ylabel(r'$l_2$')
ax.xaxis.set_label_coords(1.015, -0.03)
ax.yaxis.set_label_coords(-0.01, 1.01)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.plot(pf[:, 0], pf[:, 1], lw=3, c='grey', label='Pareto Front',zorder=0)
ax.grid(color="k", linestyle="-.", alpha=0.3,zorder=0)
plt.legend(prop=font_legend, loc='upper right')
plt.title('task preference', **font1)
# plt.legend(prop=font_legend)

if method == 'PNG':
    fig.savefig('figures/general/' + method + 'alpha_' + str(alpha) + 'thre_' + str(threshold) + '_recover.pdf')
else:
    fig.savefig('figures/general/' + method + '_recover.pdf')
plt.show()


