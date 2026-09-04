"""Arquitectura hibrida cuantico-clasica (Q-PINN) para el pendulo de Wilberforce."""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4   # qubits del registro cuantico
N_QLAYERS = 2  # capas del ansatz variacional

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def variational_circuit(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QPINN(nn.Module):
    def __init__(self, K, A, B, u0, u0_dot, v0, v0_dot):
        super().__init__()

        # Parametros adimensionales del sistema y condiciones iniciales
        self.K, self.A, self.B = K, A, B
        self.u0, self.u0_dot = u0, u0_dot
        self.v0, self.v0_dot = v0, v0_dot

        self.epsilon_causal = 0.01
        self.B_freq = nn.Parameter(torch.randn(1, 64) * 0.50, requires_grad=False)

        # Arquitectura de la Red Hibrida
        self.encoder = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, N_QUBITS), nn.Sigmoid(),  # mapea a [0,1] como angulos
        )

        weight_shapes = {"weights": (N_QLAYERS, N_QUBITS, 3)}
        self.q_layer = qml.qnn.TorchLayer(variational_circuit, weight_shapes)

        # Decoder: expansion progresiva para reconstruir z y theta
        self.decoder = nn.Sequential(
            nn.Linear(N_QUBITS, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2),
        )

    # -- Diagnostico: Neural Tangent Kernel (NTK) -----------------
    def compute_ntk_eigenvalues(self, tau_subset):
        """Calcula los autovalores del NTK empirico para la variable u."""
        J = []
        for t in tau_subset:
            uv = self.forward(t.view(1, 1))
            u = uv[:, 0]

            self.zero_grad()
            u.backward(retain_graph=True)

            grads = []
            for p in self.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            if grads:
                J.append(torch.cat(grads))

        J = torch.stack(J)  # Matriz Jacobiana
        NTK = J @ J.T       # Matriz NTK

        eigenvalues = torch.linalg.eigvalsh(NTK)
        return eigenvalues.detach().cpu().numpy()

    # -- Propagacion hacia adelante -------------------------------
    def forward(self, tau):
        t_proj = 2.0 * torch.pi * (tau @ self.B_freq)
        t_fourier = torch.cat([torch.cos(t_proj), torch.sin(t_proj)], dim=-1)

        x = self.encoder(t_fourier)
        x = self.q_layer(x)
        return self.decoder(x)

    # -- Residuos -------------------------------------------------
    def physics_residual(self, tau, use_causuality=True):
        uv = self.forward(tau)
        u, v = uv[:, :1], uv[:, 1:]

        u_t = torch.autograd.grad(u, tau, torch.ones_like(u), create_graph=True)[0]
        v_t = torch.autograd.grad(v, tau, torch.ones_like(v), create_graph=True)[0]

        u_tt = torch.autograd.grad(u_t, tau, torch.ones_like(u_t), create_graph=True)[0]
        v_tt = torch.autograd.grad(v_t, tau, torch.ones_like(v_t), create_graph=True)[0]

        f_u = u_tt + u + self.A * v
        f_v = v_tt + self.K * v + self.B * u

        loss_t = f_u**2 + f_v**2

        if use_causuality:
            loss_t_shifted = torch.cat([torch.zeros(1, 1, device=tau.device), loss_t[:-1, :]], dim=0)
            loss_cumsum = torch.cumsum(loss_t_shifted, dim=0)

            W = torch.exp(-self.epsilon_causal * loss_cumsum).detach()
            mse_physics = torch.mean(W * loss_t)
        else:
            mse_physics = torch.mean(loss_t)

        return mse_physics

    def ic_residual(self, tau0):
        uv0 = self.forward(tau0)
        u0p, v0p = uv0[:, :1], uv0[:, 1:]

        u0t = torch.autograd.grad(u0p, tau0, torch.ones_like(u0p), create_graph=True)[0]
        v0t = torch.autograd.grad(v0p, tau0, torch.ones_like(v0p), create_graph=True)[0]

        return ((self.u0 - u0p)**2 + (self.v0 - v0p)**2
                + (self.u0_dot - u0t)**2 + (self.v0_dot - v0t)**2).squeeze()

    # -- Entrenamiento (Gradient Annealing + L-BFGS) --------------
    def train(self, tau, lr_adam=1e-3, epochs_adam=2000, alpha=0.1, epochs_lbfgs=1000):
        list_loss_phy = []
        list_loss_ic = []
        list_lambda_ic = []

        list_grad_phy = []
        list_grad_ic = []
        list_ntk_eigvals = []
        ntk_epochs = []

        lambda_ic_dinamico = 100.0

        print("--- Iniciando Optimizacion con Adam ---")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_adam)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100
        )

        # Puntos para el calculo del NTK (subconjunto para no saturar memoria)
        tau_ntk = tau[::max(1, len(tau)//20)].clone().detach().requires_grad_(True)

        for epoch in range(epochs_adam):
            tau0 = torch.tensor([[0.0]], requires_grad=True)

            # 1. Extraer gradientes de la Fisica
            self.zero_grad()
            loss_phy = self.physics_residual(tau)
            loss_phy.backward(retain_graph=True)

            grads_phy = [torch.max(torch.abs(p.grad)) for p in self.parameters() if p.grad is not None]
            grad_phy_max = torch.max(torch.stack(grads_phy)) if grads_phy else torch.tensor(0.0)
            list_grad_phy.append(grad_phy_max.item())

            # 2. Extraer gradientes de las CI
            self.zero_grad()
            loss_ic = self.ic_residual(tau0)
            loss_ic.backward(retain_graph=True)

            grads_ic = [torch.mean(torch.abs(p.grad)) for p in self.parameters() if p.grad is not None]
            grad_ic_mean = torch.mean(torch.stack(grads_ic)) if grads_ic else torch.tensor(1e-8)
            list_grad_ic.append(grad_ic_mean.item())

            # 3. Gradient Annealing
            alpha_act = 0.01
            if epoch > 500:   # Warm-up
                with torch.no_grad():
                    lambda_hat_teorico = grad_phy_max / (grad_ic_mean + 1e-8)
                    lambda_hat_seguro = torch.clamp(lambda_hat_teorico, min=1.0, max=1000.0)
                    lambda_ic_dinamico = (1.0 - alpha_act) * lambda_ic_dinamico + alpha_act * lambda_hat_seguro.item()

            # 4. Actualizacion de pesos
            self.zero_grad()
            loss_total = loss_phy + (lambda_ic_dinamico * loss_ic)
            loss_total.backward()
            optimizer.step()
            scheduler.step(loss_total)

            # 5. Tracking de NTK cada 500 epocas
            if epoch % 500 == 0:
                eigvals = self.compute_ntk_eigenvalues(tau_ntk)
                list_ntk_eigvals.append(eigvals)
                ntk_epochs.append(epoch)
                print(f"Epoca {epoch:4d} | Perdida Total: {loss_total.item():.4e} | lambda_ic: {lambda_ic_dinamico:.2f}")

            list_loss_phy.append(loss_phy.item())
            list_loss_ic.append(loss_ic.item())
            list_lambda_ic.append(lambda_ic_dinamico)

        print(f"--- Cambiando a L-BFGS con lambda_ic congelado en: {lambda_ic_dinamico:.2f} ---")

        optimizer_lbfgs = torch.optim.LBFGS(self.parameters(),
                                            lr=1.0, max_iter=20, max_eval=25,
                                            tolerance_grad=1e-5, tolerance_change=1e-7,
                                            history_size=50, line_search_fn="strong_wolfe")

        lbfgs_iter = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            tau0_closure = torch.tensor([[0.0]], requires_grad=True)

            loss_phy_closure = self.physics_residual(tau, use_causuality=False)
            loss_ic_closure = self.ic_residual(tau0_closure)

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

    # -- Prediccion -----------------------------------------------
    def predict(self, tau):
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            result = self.forward(tau)
        return result.cpu().numpy()

    # -- Diagnostico ----------------------------------------------
    def print_parameter_summary(self):
        n_cls = sum(p.numel() for name, p in self.named_parameters() if 'q_layer' not in name)
        n_qnt = sum(p.numel() for name, p in self.named_parameters() if 'q_layer' in name)
        print("--- Resumen de Parametros ---")
        print(f"  Clasicos : {n_cls}")
        print(f"  Cuanticos: {n_qnt}  ({N_QLAYERS} capas x {N_QUBITS} qubits x 3 angulos)")
        print(f"  Total    : {n_cls + n_qnt}")
        print("-----------------------------\n")