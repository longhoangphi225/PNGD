import numpy as np
import torch
from tqdm import tqdm
MAX = 100000.
EPS = 1e-4

relu = torch.nn.ReLU()

class PNG_solver:
    def __init__(self, max_iters=100, n_dim=20., step_size=.1, alpha=0.5):
        self.max_iters = max_iters
        self.n_dim = n_dim
        self.step_size = step_size

    def task(self, x, grad=False):
        """
        :param x: particle set, num_particle, num_var
        :param grad: bool, whether to compute grad
        :return: val: num_particle, num_task; grad: num_task, num_particle, num_var
        """
        dim = torch.tensor(self.n_dim)
        if grad:
            # x.grad.data.zero_()
            f1 = - torch.exp(-((x - 1. / torch.sqrt(dim)) ** 2).sum(axis=1)) + 1.
            f1.sum().backward()
            grad1 = x.grad.data.detach().clone()
            x.grad.data.zero_()
            f2 = - torch.exp(-((x + 1. / torch.sqrt(dim)) ** 2).sum(axis=1)) + 1.
            f2.sum().backward()
            grad2 = x.grad.data.detach().clone()
            x.grad.data.zero_()
            return torch.stack([f1, f2]).T, torch.stack([grad1, grad2])
        else:
            f1 = - torch.exp(-((x - 1. / torch.sqrt(dim)) ** 2).sum(axis=1)) + 1.
            f2 = - torch.exp(-((x + 1. / torch.sqrt(dim)) ** 2).sum(axis=1)) + 1.
            return torch.stack([f1, f2]).T
    def task1(self, x, grad=False):
        """
        :param x: particle set, num_particle, num_var
        :param grad: bool, whether to compute grad
        :return: val: num_particle, num_task; grad: num_task, num_particle, num_var
        """
        dim = torch.tensor(self.n_dim)
        if grad:
            # x.grad.data.zero_()
            f1 = x[0][0]**2 + torch.Tensor([1/4])*(x[0][1]-torch.Tensor([9/2]))**2
            f1.sum().backward()
            grad1 = x.grad.data.detach().clone()
            x.grad.data.zero_()
            f2 = x[0][1]**2 + torch.Tensor([1/4])*(x[0][0]-torch.Tensor([9/2]))**2
            f2.sum().backward()
            grad2 = x.grad.data.detach().clone()
            x.grad.data.zero_()
            return torch.stack([f1, f2]).T, torch.stack([grad1, grad2])
        else:
            f1 = x[0][0]**2 + torch.Tensor([1/4])*(x[0][1]-torch.Tensor([9/2]))**2
            f2 = x[0][1]**2 + torch.Tensor([1/4])*(x[0][0]-torch.Tensor([9/2]))**2
            return torch.stack([f1, f2]).T

    def energy_fn(self, x, pref):
        F = self.task(x) # 1, num_task
        n_task = len(pref)
        energy = torch.zeros(n_task)
        for task_id in range(n_task):
            energy[task_id] += F[0, task_id] * pref[task_id]
        energy /= energy.sum()
        energy = (torch.log(n_task * energy) * energy).sum()

        x.grad.data.zero_()
        energy.backward()
        energy_grad = x.grad.clone().detach()

        return energy, energy_grad

    def min_norm_F(self, x):
        energy = (x**2).sum()
        x.grad.data.zero_()
        energy.backward()
        energy_grad = x.grad.clone().detach()

        return energy, energy_grad

    def template_F(self, x, context):
        F = self.task(x)
        energy = context[1] * (F[0,0] - context[0]) ** 2 + context[0] * (F[0,1] - context[1]) ** 2
        # energy = context[0] * (F[0, 0] - context[0]) ** 2 + context[1] * (F[0, 1] - context[1]) ** 2
        x.grad.data.zero_()
        energy.backward()
        energy_grad = x.grad.clone().detach()
        return energy, energy_grad

    def complex_cos_F(self, x, context):
        F = self.task(x)
        energy = - torch.cos(0.5 * 3.14159 * (F[0,0] - context[0])) * ((1. + torch.cos(3.14159 * (F[0,1] - context[1]))) ** 2)
        x.grad.data.zero_()
        energy.backward()
        energy_grad = x.grad.clone().detach()
        return energy, energy_grad

    def utility_F(self, x, context):
        F = self.task1(x)
        #energy = - torch.cos(0.5 * 3.14159 * (F[0,0] - context[0])) * ((1. + torch.cos(3.14159 * (F[0,1] - context[1]))) ** 2)
        energy = context[0]*F[0,0] + context[1]*F[0,1]
        #energy = context[0]*torch.log(F[0,0]) + context[1]*torch.log(F[0,1])
        #energy = ((F[0,0]+1)**context[0])*((F[0,1]+2)**context[1])
        x.grad.data.zero_()
        energy.backward()
        energy_grad = x.grad.clone().detach()
        return energy, energy_grad


    def project_to_linear_span(self, g, G, epsilon=1e-6):
        """
        :param g: num_particle, num_var
        :param G: num_task, num_particle, num_var
        :return: for each particle, get the projection on linear spam of G
        """
        g_project = torch.zeros_like(g)
        n_task = G.shape[0]
        n_particle = G.shape[1]
        betas = torch.zeros(n_task, n_particle)
        for _ in range(g.shape[0]):
            beta, LU = torch.solve(G[:, _, :].mm(g[[_], :].T), G[:, _, :].mm(G[:, _, :].T) + epsilon * torch.eye(G.shape[0]))
            g_project[_, :] = (beta.T).mm(G[:, _, :]).squeeze()

            betas[:, [_]] = beta

        return g_project, betas

    def mgd(self, task_grad, alpha=None): #-> return g(\theta(t))
        """
        implemented with Frank wolfe
        task_grad: num_task, num_particle, num_var
        alpha: num_task, num_particle
        """
        max_iter = 30
        #print(task_grad)
        if alpha is None:
            # initialize
            alpha = torch.zeros(task_grad.shape[0], task_grad.shape[1])
            norm = (task_grad**2).sum(axis=2) # num_task, num_particle
            id = norm.argmin(axis=0)
            alpha[id, torch.arange(0, task_grad.shape[1])] += 1.

        grad = (torch.unsqueeze(alpha, 2)*task_grad).sum(axis=0) # num_particle, num_var
        #print(grad)
        for _ in range(max_iter):
            product = (task_grad * grad).sum(axis=2) # num_task, num_particle
            id = product.argmin(axis=0) # num_task
            # selected_grad = task_grad[id, torch.arange(0, task_grad.shape[1]), :] # num_particle, num_var
            # delta = selected_grad - grad
            # todo: check lr!
            # lr = -(grad * delta).sum(axis=1) / (delta**2).sum(axis=1) # num_particle
            lr = 1./(1.+ _)
            incre = torch.zeros_like(alpha)
            incre[id, torch.arange(0, task_grad.shape[1])] += 1.
            alpha = alpha * (1. - lr) + lr * incre
            grad = (torch.unsqueeze(alpha, 2)*task_grad).sum(axis=0)
        return grad

    def grad_search_constrain(self, energy_grads, task_grads, alpha=0.5, threshold=0.01):
        """
        energy_grad: num_particle, num_var; torch.tensor -> 1x10
        task_grad: num_task, num_particle, num_var; torch.tensor ->2x1x10
        num_particle = 1
        loss: (1/2)*||grad_F+lda*grad_l||^2 - lda*phi
        """

        n_particle = task_grads.shape[1]
        #threshold = 1e-10
        # obtain mgd gradient
        d_mgd = self.mgd(task_grads) # g(\theta(t))
        d_mgd_norm = (d_mgd ** 2).sum(axis=1)
        # constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold)
        constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold * (task_grads**2).sum(axis=2).mean()) # \phi(t): shape[1]

        betas_mask = (d_mgd_norm > threshold).unsqueeze(0).unsqueeze(2) # 1x1x1

        grad_project, betas = self.project_to_linear_span(-energy_grads, task_grads)  # init beta; betas: n_task, n_particle ->2x1x1
        betas = betas * (betas >= 0.)
        betas = betas.unsqueeze(2)
        betas.requires_grad = True
        lr = self.step_size
        for _ in range(200):
            loss = 0.5 * (((task_grads * betas).sum(axis=0) + energy_grads) ** 2).sum() / n_particle \
                   - (betas.sum(axis=0).squeeze() * constrain).sum()
            
            loss.backward()
            betas.data -= lr * betas.grad.data
            betas.grad.data.zero_()
            betas.data *= (betas.data >= 0.)
        grad_constrain = ((task_grads * (betas * betas_mask)).sum(axis=0) + energy_grads).detach().clone() # v(t): 1x10
        return grad_constrain, betas * betas_mask
    def grad_search_constrain1(self, energy_grads, task_grads, alpha=0.5, threshold=0.1, C = 1):
        """
        energy_grad: num_particle, num_var; torch.tensor 
        task_grad: num_task, num_particle, num_var; torch.tensor
        num_particle = 1
        loss: <grad_F,v>/||v|| + C*sum(max(g_i,0))
        """

        # obtain mgd gradient
        d_mgd = self.mgd(task_grads) # g(\theta(t))
        d_mgd_norm = (d_mgd ** 2).sum(axis=1)
        constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold * (task_grads**2).sum(axis=2).mean()) # \phi(t)

        betas_mask = (d_mgd_norm > threshold).unsqueeze(0).unsqueeze(2)

        _, betas = self.project_to_linear_span(-energy_grads, task_grads)  # init beta; betas: n_task, n_particle
        betas = betas * (betas >= 0.)
        betas = betas.unsqueeze(2)
        v_init = torch.rand(energy_grads.shape[0],energy_grads.shape[1])
        v_init.requires_grad = True
        lr = self.step_size
        for _ in range(200):
            hinge = constrain - torch.matmul(task_grads,(v_init.T))
            cos_similar = ((energy_grads*v_init).sum())/(torch.sqrt(torch.sum(torch.square(v_init))))
            loss = ((-1) * cos_similar)+ (C)*torch.sum(torch.max(torch.zeros_like(hinge),hinge))
            loss.backward()
            v_init.data -= lr * v_init.grad.data
            v_init.grad.data.zero_()
        grad_constrain = v_init
        return grad_constrain, betas * betas_mask
    def grad_search_constrain2(self, energy_grads, task_grads, alpha=0.5, threshold=0.01, C = 1):
        """
        energy_grad: num_particle, num_var; torch.tensor
        task_grad: num_task, num_particle, num_var; torch.tensor
        num_particle = 1
        loss: ||grad_F-v||^2 + C*sum(max(g_i,0))
        """

        # obtain mgd gradient
        d_mgd = self.mgd(task_grads) # g(\theta(t))
        d_mgd_norm = (d_mgd ** 2).sum(axis=1)
        constrain = alpha * d_mgd_norm * (d_mgd_norm > threshold * (task_grads**2).sum(axis=2).mean()) # \phi(t)

        betas_mask = (d_mgd_norm > threshold).unsqueeze(0).unsqueeze(2)

        _, betas = self.project_to_linear_span(-energy_grads, task_grads)  # init beta; betas: n_task, n_particle
        betas = betas * (betas >= 0.)
        betas = betas.unsqueeze(2)
        v_init = torch.rand(energy_grads.shape[0],energy_grads.shape[1])
        v_init.requires_grad = True
        lr = self.step_size
        for _ in range(100):
            hinge = constrain - torch.matmul(task_grads,(v_init.T))
            norm2 = torch.sum(torch.square(energy_grads-v_init))
            loss = ((1/2) * norm2)+ (C)*torch.sum(torch.max(torch.zeros_like(hinge),hinge))
            loss.backward()
            v_init.data -= lr * v_init.grad.data
            v_init.grad.data.zero_()
        grad_constrain = v_init
        return grad_constrain, betas * betas_mask

    def optimize_epo(self, x, r, alpha=0.5, threshold=0.1):
        ls = []
        lr = self.step_size
        iters = self.max_iters

        for _ in range(iters):
            if _ % 20 == 0:
                print('iter: ', _)
            f, df = self.task(x, grad=True)
            e, de = self.energy_fn(x, pref=r)
            d, betas = self.grad_search_constrain(de, df, alpha=alpha, threshold=threshold)

            ls.append(f.clone().detach().squeeze().numpy())
            x.data -= lr * d

        res = {'ls': np.stack(ls)}
        return x.clone().detach().squeeze().numpy(), res

    def optimize(self, x, criterion='min norm', context=None, alpha=0.5, threshold=0.1, start=-1):
        ls = []
        lr = self.step_size
        iters = self.max_iters
        for _ in range(iters):

            f, df = self.task1(x, grad=True)
            if _ <= start:
                d = self.mgd(df)
            else:
                if criterion == 'min norm':
                    e, de = self.min_norm_F(x)
                elif criterion == 'complex sin':
                    e, de = self.complex_sin_F(x)
                elif criterion == 'template':
                    e, de = self.template_F(x, context)
                elif criterion == 'complex cos':
                    e, de = self.complex_cos_F(x, context)
                elif criterion == 'utility':
                    e, de = self.utility_F(x, context)
                d, betas = self.grad_search_constrain(de, df, alpha=alpha, threshold=threshold)
            ls.append(f.clone().detach().squeeze().numpy())
            x.data -= lr * d

        res = {'ls': np.stack(ls)}
        return x.clone().detach().squeeze().numpy(), res
    def optimize1(self, x, criterion='min norm', context=None, alpha=0.5, threshold=0.1, start=-1,C=1):
        ls = []
        lr = self.step_size
        iters = self.max_iters
        for _ in range(iters):

            f, df = self.task1(x, grad=True)
            if _ <= start:
                d = self.mgd(df)
            else:
                if criterion == 'min norm':
                    e, de = self.min_norm_F(x)
                elif criterion == 'complex sin':
                    e, de = self.complex_sin_F(x)
                elif criterion == 'template':
                    e, de = self.template_F(x, context)
                elif criterion == 'complex cos':
                    e, de = self.complex_cos_F(x, context)
                elif criterion == 'utility':
                    e, de = self.utility_F(x, context)
                d, betas = self.grad_search_constrain1(de, df, alpha=alpha, threshold=threshold,C=C)
            ls.append(f.clone().detach().squeeze().numpy())
            x.data -= lr * d

        res = {'ls': np.stack(ls)}
        return x.clone().detach().squeeze().numpy(), res
