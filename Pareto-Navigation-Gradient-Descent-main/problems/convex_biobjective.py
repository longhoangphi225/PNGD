import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLines   # , labelLine,
from latex_utils import latexify


def f1(x):
    return x[0]**2 + (1/4)*(x[1]-(9/2))**2


def f2(x):
    return x[1]**2 + (1/4)*(x[0]-(9/2))**2


# calculate the gradients using autograd
f1_dx = grad(f1)
f2_dx = grad(f2)


def concave_fun_eval(x):
    """
    return the function values and gradient values
    """
    return [f1(x), f2(x)]


# ### create the ground truth Pareto front ###
def concave_fun_eval1(x):
    """
    return the function values and gradient values
    """
    return [9*x/(2*x-8),9/(2-8*x)]
def concave_fun_eval2(x):
    """
    return the function values and gradient values
    """
    return [x,2-x]
def concave_fun_eval3(x):
    """
    return the function values and gradient values
    """
    return [9/(2-8*x),9*x/(2*x-8)]

def create_pf1():
    ps1 = np.linspace(-1 / 2, 0, num=500)
    ps3 = np.linspace(-1 / 2, 0, num=500)
    ps2 = np.linspace(1/2, 3/2, num=500)
    pf = []
    for x1 in ps1:
        x = concave_fun_eval1(x1)
        f = concave_fun_eval(x)
        #print(f)
        pf.append(f)
    for x2 in ps2:
        x = concave_fun_eval2(x2)
        f = concave_fun_eval(x)
        pf.append(f)    
    for x3 in ps3:
        x = concave_fun_eval3(x3)
        f = concave_fun_eval(x)
        pf.append(f)     
    pf = np.array(pf)
    return pf


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def add_interval(ax, xdata, ydata,
                 color="k", caps="  ", label='', side="both", lw=2):
    line = ax.add_line(mpl.lines.Line2D(xdata, ydata))
    line.set_label(label)
    line.set_color(color)
    line.set_linewidth(lw)
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 12,
        'color': line.get_color()
    }
    a = []
    if side in ["left", "both"]:
        a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), zorder=2, **anno_args)
        a.append(a0)
    if side in ["right", "both"]:
        a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), zorder=2, **anno_args)
        a.append(a1)
    return (line, tuple(a))



