"""Arquitectura y entrenamiento de la PINN clasica."""

import numpy as np
import torch
import torch.nn as nn


class PINN:
    def __init__(self, K, A, B, u0, u0_dot, v0, v0_dot):
        # Parametros Adimensionales
        self.K = K
        self.A = A
        self.B = B

        # Condiciones iniciales Adimensionales
        self.u0 = u0
        self.u0_dot = u0_dot
        self.v0 = v0
        self.v0_dot = v0_dot

        # Hiperparametro Causal
        self.epsilon_causal = 0.01

        self.B_freq = nn.Parameter(torch.randn(1, 64) * 0.5, requires_grad=False)

        # Arquitectura de la Red Neuronal (Pesos y Sesgos)
        self.w0 = nn.Parameter(torch.randn(128, 128) * np.sqrt(2 / (128 + 128)))
        self.b0 = nn.Parameter(torch.zeros(1, 128))
        self.w1 = nn.Parameter(torch.randn(128, 128) * np.sqrt(2 / (128 + 128)))
        self.b1 = nn.Parameter(torch.zeros(1, 128))
        self.w2 = nn.Parameter(torch.randn(128, 128) * np.sqrt(2 / (128 + 128)))
        self.b2 = nn.Parameter(torch.zeros(1, 128))
        self.w3 = nn.Parameter(torch.randn(128, 128) * np.sqrt(2 / (128 + 128)))
        self.b3 = nn.Parameter(torch.zeros(1, 128))
        self.w4 = nn.Parameter(torch.randn(128, 128) * np.sqrt(2 / (128 + 128)))
        self.b4 = nn.Parameter(torch.zeros(1, 128))
        self.w5 = nn.Parameter(torch.randn(128, 128) * np.sqrt(2 / (128 + 128)))
        self.b5 = nn.Parameter(torch.zeros(1, 128))
        self.w6 = nn.Parameter(torch.randn(128, 128) * np.sqrt(2 / (128 + 128)))
        self.b6 = nn.Parameter(torch.zeros(1, 128))
        self.w7 = nn.Parameter(torch.randn(128, 2) * np.sqrt(2 / (2 + 128)))
        self.b7 = nn.Parameter(torch.zeros(1, 2))

        self.params = [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2,
                       self.w3, self.b3, self.w4, self.b4, self.w5, self.b5,
                       self.w6, self.b6, self.w7, self.b7]

    def contar_parametros(self):
        return sum(p.numel() for p in self.params)

    def tanh(self, x):
        return torch.tanh(x)

    def compute_ntk_eigenvalues(self, tau_subset):
        """Calcula los autovalores del NTK para monitorear el sesgo espectral."""
        J = []
        for t in tau_subset:
            uv = self.forward_propagation(t.view(1, 1))
            u = uv[:, 0]

            # Limpiamos los gradientes manualmente (no es un nn.Module)
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

            u.backward(retain_graph=True)

            grads = []
            for p in self.params:
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            J.append(torch.cat(grads))

        J = torch.stack(J)  # Matriz Jacobiana
        NTK = J @ J.T       # Matriz NTK empirica

        eigenvalues = torch.linalg.eigvalsh(NTK)
        return eigenvalues.detach().cpu().numpy()

    def forward_propagation(self, t):
        t_proj = 2.0 * torch.pi * (t @ self.B_freq)
        t_fourier = torch.cat([torch.cos(t_proj), torch.sin(t_proj)], dim=-1)

        s0 = t_fourier @ self.w0 + self.b0
        a0 = self.tanh(s0)
        s1 = a0 @ self.w1 + self.b1
        a1 = self.tanh(s1)
        s2 = a1 @ self.w2 + self.b2
        a2 = self.tanh(s2)
        s3 = a2 @ self.w3 + self.b3
        a3 = self.tanh(s3)
        s4 = a3 @ self.w4 + self.b4
        a4 = self.tanh(s4)
        s5 = a4 @ self.w5 + self.b5
        a5 = self.tanh(s5)
        s6 = a5 @ self.w6 + self.b6
        a6 = self.tanh(s6)
        s7 = a6 @ self.w7 + self.b7
        return s7

    def get_physics_residual(self, tau, use_causality=True):
        uv_pred = self.forward_propagation(tau)
        u = uv_pred[:, 0:1]
        v = uv_pred[:, 1:2]

        u_dot = torch.autograd.grad(u, tau, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_dot = torch.autograd.grad(v, tau, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        u_ddot = torch.autograd.grad(u_dot, tau, grad_outputs=torch.ones_like(u_dot), create_graph=True)[0]
        v_ddot = torch.autograd.grad(v_dot, tau, grad_outputs=torch.ones_like(v_dot), create_graph=True)[0]

        loss_t = (u_ddot + u + self.A * v)**2 + (v_ddot + self.K * v + self.B * u)**2

        if use_causality:
            loss_t_shifted = torch.cat([torch.zeros(1, 1, device=tau.device), loss_t[:-1, :]], dim=0)
            loss_cumsum = torch.cumsum(loss_t_shifted, dim=0)
            W = torch.exp(-self.epsilon_causal * loss_cumsum).detach()
            mse_physics = torch.mean(W * loss_t)
        else:
            mse_physics = torch.mean(loss_t)

        return mse_physics

    def get_ic_residual(self, tau0):
        uv_0 = self.forward_propagation(tau0)
        u_pred0 = uv_0[:, 0:1]
        v_pred0 = uv_0[:, 1:2]

        u_dot_pred0 = torch.autograd.grad(u_pred0, tau0, grad_outputs=torch.ones_like(u_pred0), create_graph=True)[0]
        v_dot_pred0 = torch.autograd.grad(v_pred0, tau0, grad_outputs=torch.ones_like(v_pred0), create_graph=True)[0]

        loss_u0 = (self.u0 - u_pred0)**2
        loss_v0 = (self.v0 - v_pred0)**2
        loss_u0_dot = (self.u0_dot - u_dot_pred0)**2
        loss_v0_dot = (self.v0_dot - v_dot_pred0)**2

        ic_residual = loss_u0 + loss_v0 + loss_u0_dot + loss_v0_dot
        return ic_residual.squeeze()

    def train(self, tau, lr_adam, epochs_adam, alpha=0.1, epochs_lbfgs=1000):
        list_loss_phy = []
        list_loss_ic = []
        list_lambda_ic = []

        list_grad_phy = []
        list_grad_ic = []
        list_ntk_eigvals = []
        ntk_epochs = []

        lambda_ic_dinamico = 10.0

        print("--- Iniciando Optimizacion con Adam ---")
        optimizer = torch.optim.Adam(self.params, lr=lr_adam)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

        # Subconjunto de puntos para el calculo del NTK
        tau_ntk = tau[::max(1, len(tau)//20)].clone().detach().requires_grad_(True)

        for epoch in range(epochs_adam):
            tau0 = torch.tensor([[0.0]], requires_grad=True)

            # 1. Gradientes de la Fisica
            optimizer.zero_grad()
            loss_phy = self.get_physics_residual(tau)
            loss_phy.backward(retain_graph=True)
            grad_phy_max = torch.max(torch.stack([torch.max(torch.abs(p.grad)) for p in self.params if p.grad is not None]))
            list_grad_phy.append(grad_phy_max.item())

            # 2. Gradientes de las Condiciones Iniciales
            optimizer.zero_grad()
            loss_ic = self.get_ic_residual(tau0)
            loss_ic.backward(retain_graph=True)
            grad_ic_mean = torch.mean(torch.stack([torch.mean(torch.abs(p.grad)) for p in self.params if p.grad is not None]))
            list_grad_ic.append(grad_ic_mean.item())

            # 3. Gradient Annealing
            alpha_dinamico = 0.01
            if epoch > 500:
                with torch.no_grad():
                    lambda_hat_teorico = grad_phy_max / (grad_ic_mean + 1e-8)
                    lambda_hat_seguro = torch.clamp(lambda_hat_teorico, min=1.0, max=200.0)
                    lambda_ic_dinamico = (1.0 - alpha_dinamico) * lambda_ic_dinamico + alpha_dinamico * lambda_hat_seguro.item()

            # 4. Actualizacion real de los pesos
            optimizer.zero_grad()
            loss_total = loss_phy + (lambda_ic_dinamico * loss_ic)
            loss_total.backward()
            optimizer.step()
            scheduler.step(loss_total)

            # 5. Tracking de NTK cada 500 epocas
            if epoch % 500 == 0:
                eigvals = self.compute_ntk_eigenvalues(tau_ntk)
                list_ntk_eigvals.append(eigvals)
                ntk_epochs.append(epoch)
                print(f"Epoca {epoch} | Loss Total: {loss_total.item():.4e} | lambda_ic: {lambda_ic_dinamico:.2f}")

            list_loss_phy.append(loss_phy.item())
            list_loss_ic.append(loss_ic.item())
            list_lambda_ic.append(lambda_ic_dinamico)

        print(f"--- Cambiando a L-BFGS con lambda_ic congelado en: {lambda_ic_dinamico:.2f} ---")

        optimizer_lbfgs = torch.optim.LBFGS(self.params,
                                            lr=1.0,
                                            max_iter=20,
                                            max_eval=25,
                                            tolerance_grad=1e-5,
                                            tolerance_change=1e-7,
                                            history_size=50,
                                            line_search_fn="strong_wolfe")

        lbfgs_iter = [0]

        def closure():
            # Limpiar gradientes manualmente para L-BFGS
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

            tau0_closure = torch.tensor([[0.0]], requires_grad=True)

            loss_phy_closure = self.get_physics_residual(tau, use_causality=False)
            loss_ic_closure = self.get_ic_residual(tau0_closure)

            loss_total_closure = loss_phy_closure + (lambda_ic_dinamico * loss_ic_closure)
            loss_total_closure.backward()

            if lbfgs_iter[0] % 100 == 0:
                print(f"L-BFGS Iter {lbfgs_iter[0]:4d} | Perdida Total: {loss_total_closure.item():.4e}")

            lbfgs_iter[0] += 1
            list_loss_phy.append(loss_phy_closure.item())
            list_loss_ic.append(loss_ic_closure.item())

            return loss_total_closure

        for epoch in range(epochs_lbfgs):
            optimizer_lbfgs.step(closure)

        return list_loss_ic, list_loss_phy, list_lambda_ic, list_grad_ic, list_grad_phy, list_ntk_eigvals, ntk_epochs

    def predict(self, tau):
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            result = self.forward_propagation(tau)
        return result.numpy()