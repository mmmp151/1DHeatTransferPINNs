import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import tensorflow as tf
import time as timelib
import pandas as pd
from pandas import DataFrame as D
from scipy.optimize import minimize

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

x = samples[:, 0] / xm
t = samples[:, 1] / tm

x = tf.convert_to_tensor(x, dtype=tf.float32)
t = tf.convert_to_tensor(t, dtype=tf.float32)
x = tf.reshape(x, (-1, 1))
t = tf.reshape(t, (-1, 1))

# Define the domain intervals for each dimension
xlimits = np.array([[0.0, xm], [0.0, 0.0]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 10000
samples = sampling(num_samples)

x_init = samples[:, 0] / xm
t_init = samples[:, 1] / tm

x_init = tf.convert_to_tensor(x_init, dtype=tf.float32)
t_init = tf.convert_to_tensor(t_init, dtype=tf.float32)
x_init = tf.reshape(x_init, (-1, 1))
t_init = tf.reshape(t_init, (-1, 1))

# Define the domain intervals for each dimension
xlimits = np.array([[0.0, 0.0], [0.0, tm]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 10000
samples = sampling(num_samples)

x_bl = samples[:, 0] / xm
t_bl = samples[:, 1] / tm

x_bl = tf.convert_to_tensor(x_bl, dtype=tf.float32)
t_bl = tf.convert_to_tensor(t_bl, dtype=tf.float32)
x_bl = tf.reshape(x_bl, (-1, 1))
t_bl = tf.reshape(t_bl, (-1, 1))

# Define the domain intervals for each dimension
xlimits = np.array([[xm, xm], [0.0, tm]])

# Create an LHS sampling instance
sampling = LHS(xlimits=xlimits)

# Generate 10000 samples
num_samples = 10000
samples = sampling(num_samples)

x_br = samples[:, 0] / xm
t_br = samples[:, 1] / tm

x_br = tf.convert_to_tensor(x_br, dtype=tf.float32)
t_br = tf.convert_to_tensor(t_br, dtype=tf.float32)
x_br = tf.reshape(x_br, (-1, 1))
t_br = tf.reshape(t_br, (-1, 1))

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x.numpy(), t.numpy(), c="b", s=1)
ax.scatter(x_bl.numpy(), t_bl.numpy(), c="r", s=1)
ax.scatter(x_br.numpy(), t_br.numpy(), c="r", s=1)
ax.scatter(x_init.numpy(), t_init.numpy(), c="k", s=1)
ax.set_aspect("equal")


class PINN(tf.keras.Model):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.losses_dict = {"loss": [], "wall": [], "initial": [], "pde": []}
        self.layers_list = []
        for i in range(len(layers) - 2):
            self.layers_list.append(
                tf.keras.layers.Dense(layers[i + 1], activation="tanh")
            )
        self.layers_list.append(tf.keras.layers.Dense(layers[-1], activation="tanh"))

    def call(self, inputs):
        x, t = inputs
        X = tf.concat([x, t], axis=1)
        for layer in self.layers_list:
            X = layer(X)
        return X

    def PDE_loss(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            T = self.call((x, t))
            T_x = tape.gradient(T, x)
        T_xx = tape.gradient(T_x, x)
        T_t = tape.gradient(T, t)
        del tape
        return tf.reduce_mean(tf.square(T_t - tm * alpha / xm**2 * T_xx))

    def bi_loss(self, l, r, T_bi):
        T = self.call((l, r))
        return tf.reduce_mean(tf.square(T - T_bi))

    def plot_losses(self):
        fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 6))
        axes[0].set_yscale("log")
        for i, j in zip(range(4), ["loss", "Wall", "initial", "pde"]):
            axes[i].plot(self.losses_dict[j.lower()])
            axes[i].set_title(j)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()


layers = [2, 8, 8, 8, 8, 8, 1]
pde_solver = PINN(layers)

# Build the model by making an initial call
pde_solver((x, t))


# Custom training loop with Adam optimizer
def train_adam(epochs, lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss_b = pde_solver.bi_loss(x_bl, t_bl, T_bl) + pde_solver.bi_loss(
                x_br, t_br, T_br
            )
            loss_init = pde_solver.bi_loss(x_init, t_init, T_init)
            loss_eq = pde_solver.PDE_loss(x, t)
            loss = loss_b + loss_eq + loss_init
        gradients = tape.gradient(loss, pde_solver.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pde_solver.trainable_variables))

        pde_solver.losses_dict["wall"].append(loss_b.numpy())
        pde_solver.losses_dict["initial"].append(loss_init.numpy())
        pde_solver.losses_dict["pde"].append(loss_eq.numpy())
        pde_solver.losses_dict["loss"].append(loss.numpy())

        if epoch % 100 == 0:
            print(
                f"Adam Epoch {epoch}, Loss: {loss.numpy()}, Boundary Loss: {loss_b.numpy()}, Initial Loss: {loss_init.numpy()}, PDE Loss: {loss_eq.numpy()}"
            )


# Custom training loop with L-BFGS optimizer
def loss_fn():
    loss_b = pde_solver.bi_loss(x_bl, t_bl, T_bl) + pde_solver.bi_loss(x_br, t_br, T_br)
    loss_init = pde_solver.bi_loss(x_init, t_init, T_init)
    loss_eq = pde_solver.PDE_loss(x, t)
    loss = loss_b + loss_eq + loss_init
    return loss


def get_weights():
    weights = [tf.reshape(var, [-1]).numpy() for var in pde_solver.trainable_variables]
    if not weights:
        raise ValueError(
            "No trainable variables in the model. Ensure the model is properly built."
        )
    return np.concatenate(weights, axis=0)


def set_weights(weights):
    idx = 0
    for var in pde_solver.trainable_variables:
        var_shape = var.shape
        var_size = tf.size(var).numpy()
        new_value = tf.convert_to_tensor(
            weights[idx : idx + var_size].reshape(var_shape), dtype=tf.float32
        )
        var.assign(new_value)
        idx += var_size


def lbfgs_optimizer():
    init_params = get_weights()

    def loss_and_grads(params):
        set_weights(params)
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, pde_solver.trainable_variables)
        grads = np.concatenate(
            [tf.reshape(grad, [-1]).numpy() for grad in grads], axis=0
        )
        return loss.numpy().astype(np.float64), grads.astype(np.float64)

    result = minimize(
        loss_and_grads,
        init_params,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    set_weights(result.x)
    return result


# Train the model using Adam optimizer
train_adam(epochs=2000, lr=5e-4)

# Train the model using L-BFGS optimizer
result = lbfgs_optimizer()

pde_solver.plot_losses()

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
l = tf.linspace(0.0, 1.0, 600)[:, tf.newaxis]
r = tf.ones_like(l) * a
y = pde_solver.call((l, r)) * Tm
print("Predicted temp at t = 3s", y)

error = abs((y.numpy() - T_N[-1]) / (T_N[-1]))
print(np.max(error * 100))
error = D(error)

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(error * 100)

a = 0.1
l = tf.linspace(0.0, 1.0, 100)[:, tf.newaxis]
r = tf.ones_like(l) * a
y = pde_solver.call((l, r)) * Tm
plt.plot(l.numpy() * xm, y.numpy())
plt.title(f"temp distribution at t = {a * tm}")
plt.show()
