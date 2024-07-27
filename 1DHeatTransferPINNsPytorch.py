import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import torch
import torch.nn as nn
import time as timelib
import pandas as pd
from pandas import DataFrame as D

device = torch.device(0)


st = timelib.time()

alpha = 1e-5
xm = 7e-2
tm = 5
T_bl = 400.0
T_br = 500.0
T_init = 300.0

Tm = max(T_bl, T_br, T_init)

T_bl = T_bl / Tm
T_br = T_br / Tm
T_init = T_init / Tm

# Define the domain intervals for each dimension
xlimits = np.array([[0.0, xm], [0.0, tm]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 10000
samples = sampling(num_samples)


# x = np.ones_like(t) * 0.0
x = samples[:, 0] / xm
t = samples[:, 1] / tm
x = torch.from_numpy(x)
t = torch.from_numpy(t)
x = x.float().to(device)
t = t.float().to(device)
x.requires_grad = True
t.requires_grad = True

# Define the domain intervals for each dimension
xlimits = np.array([[0.0, xm], [0.0, 0.0]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 10000
samples = sampling(num_samples)

x_init = samples[:, 0] / xm
t_init = samples[:, 1] / tm
x_init = torch.from_numpy(x_init)
t_init = torch.from_numpy(t_init)
x_init = x_init.float().to(device)
t_init = t_init.float().to(device)
x_init.requires_grad = True
t_init.requires_grad = True

# Define the domain intervals for each dimension
xlimits = np.array([[0.0, 0.0], [0.0, tm]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 10000
samples = sampling(num_samples)

x_bl = samples[:, 0] / xm
t_bl = samples[:, 1] / tm
x_bl = torch.from_numpy(x_bl)
t_bl = torch.from_numpy(t_bl)
x_bl = x_bl.float().to(device)
t_bl = t_bl.float().to(device)
x_bl.requires_grad = True
t_bl.requires_grad = True

# Define the domain intervals for each dimension
xlimits = np.array([[xm, xm], [0.0, tm]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 10000
samples = sampling(num_samples)

x_br = samples[:, 0] / xm
t_br = samples[:, 1] / tm
x_br = torch.from_numpy(x_br)
t_br = torch.from_numpy(t_br)
x_br = x_br.float().to(device)
t_br = t_br.float().to(device)
x_br.requires_grad = True
t_br.requires_grad = True


fig = plt.figure()
ax = fig.add_subplot()
print(len(x.cpu().detach().numpy()))
ax.scatter(x.cpu().detach().numpy(), t.cpu().detach().numpy(), c="b", s=1)
ax.scatter(x_bl.cpu().detach().numpy(), t_bl.cpu().detach().numpy(), c="r", s=1)
ax.scatter(x_br.cpu().detach().numpy(), t_br.cpu().detach().numpy(), c="r", s=1)
ax.scatter(x_init.cpu().detach().numpy(), t_init.cpu().detach().numpy(), c="k", s=1)
ax.set_aspect("equal")


class PINN(nn.Module):
    def __init__(self, layers) -> None:
        super(PINN, self).__init__()

        self.losses = {"loss": [], "wall": [], "initial": [], "pde": []}
        self.dropout_prob = 1e-10

        self.layers = layers
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1]))
            self.net.add_module(f"activation_{i}", nn.Tanh())
            # self.net.add_module(f'dropout_{i}', nn.Dropout(p=self.dropout_prob))  # Add dropout
        self.net.add_module("output", nn.Linear(layers[-2], layers[-1]))
        self.net.add_module(f"activation_output", nn.Tanh())

        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4, weight_decay=1e-4)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=2000,
            max_eval=2000,
            tolerance_grad=0,
            tolerance_change=0,
            history_size=500,
            line_search_fn="strong_wolfe",
        )

    def forward(self, x, t):
        X = torch.cat([x.view(-1, 1), t.view(-1, 1)], axis=1)
        u = self.net(X)
        return u[:, 0]

    def PDE_loss(self):
        T = self.forward(x, t)
        T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0]
        T_t = torch.autograd.grad(T.sum(), t, create_graph=True)[0]
        return torch.mean(torch.square(T_t - tm * alpha / xm**2 * T_xx))

    def bi_loss(self, l, r, T_bi):
        T = self.forward(l, r)
        return torch.mean(torch.square(T - T_bi))

    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        loss_b = self.bi_loss(x_bl, t_bl, T_bl) + self.bi_loss(x_br, t_br, T_br)
        loss_init = self.bi_loss(x_init, t_init, T_init)
        loss_eq = self.PDE_loss()

        self.losses["wall"].append(loss_b.detach().cpu().item())
        self.losses["initial"].append(loss_init.detach().cpu().item())
        self.losses["pde"].append(loss_eq.detach().cpu().item())

        loss = loss_b + loss_eq + loss_init
        self.losses["loss"].append(loss.detach().cpu().item())

        print(
            f"\r epoch {len(self.losses['pde'])} , loss : {loss.detach().cpu().item():5e} , loss boundary : {loss_b.detach().cpu().item():5e} , loss initial : {loss_init.detach().cpu().item():5e} , loss PDE : {loss_eq.detach().cpu().item():5e} , time : {timelib.time()-st:.2f} s",
            end="",
        )
        if len(self.losses["pde"]) % 100 == 0:
            print("")

        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)

        return loss

    def train(self, optimizer, epoch):
        try:
            c = 0
            for i in optimizer:
                if i == self.adam:
                    print("")
                    print("\noptimizer : ADAM")
                else:
                    print("")
                    print("\noptimizer : LBFGS")
                for j in range(epoch[c]):
                    if i == self.adam:
                        ls = self.closure()
                        i.step()
                    else:
                        i.step(self.closure)

                c += 1
        except KeyboardInterrupt:
            print("")
            print("intrrupted by user")

    def plot(self):
        with torch.no_grad():
            fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 6))
            axes[0].set_yscale("log")
            for i, j in zip(range(4), ["loss", "Wall", "initial", "pde"]):
                axes[i].plot(self.losses[j.lower()])
                axes[i].set_title(j)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")


pde_solver = PINN([2, 8, 8, 8, 8, 8, 1]).to(device)
optimizer = [pde_solver.adam, pde_solver.lbfgs]
epoch = [2000, 200]


pde_solver.train(optimizer, epoch)


pde_solver.plot()


# Calculation of numerical solution for 3s
N_x = 600
dx = 7e-2 / N_x
dt = 0.5 * dx**2 / alpha
N_t = int(3.0 / dt)
T_prime = [[T_init * Tm] * N_x] * N_t
T_N = np.array(T_prime, dtype=np.float32)
r = alpha * dt / dx**2
T_N[:, 0] = T_bl * Tm
T_N[:, -1] = T_br * Tm
print(N_t)
for t_a in range(1, N_t):
    for x_a in range(1, N_x - 1):
        T_N[t_a, x_a] = T_N[t_a - 1, x_a] + r * (
            T_N[t_a - 1, x_a + 1] - 2 * T_N[t_a - 1, x_a] + T_N[t_a - 1, x_a - 1]
        )

print("Numerical solution at t = 3s :", T_N[-1])

# Checking the network output with numerical solution
a = 0.6
l = torch.linspace(0, 1, 600).to(device)
r = torch.ones_like(l) * a
y = pde_solver(l, r) * Tm
print("Predicted temp at t = 3s", y)


error = abs((y.detach().cpu().numpy() - T_N[-1]) / (T_N[-1]))
print(max(error * 100))
error = D(error)


with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(error * 100)


a = 0.1
l = torch.linspace(0, 1, 100).to(device)
r = torch.ones_like(l) * a
y = pde_solver(l, r) * Tm
# l = torch.linspace(0,a*xm,1000).to(device)
# y_ex = exact_sol(l)
plt.plot(l.detach().cpu().numpy() * xm, y.detach().cpu().numpy())
# plt.plot(l.detach().cpu().numpy(),y_ex.detach().cpu().numpy())
plt.title(f"temp distribution at t = {a*tm}")
plt.show()
