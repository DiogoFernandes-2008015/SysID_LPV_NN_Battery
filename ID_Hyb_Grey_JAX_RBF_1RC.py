from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.io import loadmat
from tqdm import tqdm  # For the progress bar
import os
import scipy.io
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, LinearInterpolation
from jax import random, grad, jit, vmap
from jax.flatten_util import ravel_pytree

# Checking where JAX is running
try:
    print(f"JAX is running on: {jax.devices()[0].platform.upper()}")
except IndexError:
    print("No JAX devices found.")

jax.config.update("jax_enable_x64", True)

# Loading training dataset
DATA = loadmat('data_train.mat')
u = DATA['i']
y = DATA['v']
time = DATA['t']

fig, axs = plt.subplots(2, 1, sharex=True)  # sharex makes sense for time series

# Plot 1: Input u
axs[0].plot(time, u, color='b')  # Added color for clarity
axs[0].set_title('Input Signal (u) vs. Time')
axs[0].set_ylabel('u (Input)')
axs[0].grid(True)

# Plot 2: Output y and Reference yref
axs[1].plot(time, y, 'k', label='y (Output)')
axs[1].set_title('Output (y) vs. Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()  # Adjusts subplot params for a tight layout
plt.show()

# Decimation of the signals
decimate = 1
u = u[::decimate]
y = y[::decimate]
time = time[::decimate]

if time is not None:
    time = time.flatten()
    u = u.flatten()
    y = y.flatten()

    N = time.shape[0]
    Ts = time[1] - time[0]
    fs = 1 / Ts
    T = time[-1]
    print(f"Training Dataset\nN={N}, fs={fs}, T={T}, Ts={Ts}")

# --- Setup for Multiple Shooting ---
n_shots = 10 # random
n_timesteps_per_shot = N // n_shots
n_states = 2


def init_rbf_params(input_size, num_rbf_neurons, output_size, key, scale=1e-2):
    """
    Inicializa os parâmetros da Rede Neural de Base Radial (RBFN).
    Parâmetros: Centros (C), Larguras (Sigma) e Pesos de Saída (W_out, b_out).
    """
    c_key, sigma_key, w_key, b_key = random.split(key, 4)

    # 1. Centros (C): [num_rbf_neurons, input_size].
    # Inicializados aleatoriamente
    C = random.uniform(c_key, (num_rbf_neurons, input_size), minval=0.0, maxval=1.0)

    # 2. Larguras (Sigma): [num_rbf_neurons].
    # Inicializadas com um valor constante pequeno ou aleatório
    log_sigma = random.normal(sigma_key, (num_rbf_neurons,))

    # 3. Pesos de Saída (W_out): [output_size, num_rbf_neurons]
    W_out = scale * random.normal(w_key, (output_size, num_rbf_neurons))

    # 4. Bias de Saída (b_out): [output_size]
    b_out = scale * random.normal(b_key, (output_size,))

    return [C, log_sigma, W_out, b_out]


def gaussian_rbf(x, c, log_sigma):
    """Função de Base Radial Gaussiana."""
    # x é o input (e.g., [SOC]), c é o centro, exp(log_sigma) é o desvio padrão
    sigma = jnp.exp(log_sigma) + 1e-6  # Adiciona um epsilon para evitar zero
    dist_sq = jnp.sum((x - c) ** 2)
    # Note que no RBF, o input x pode ser [1] (SOC), mas c é [num_rbf_neurons, 1]
    # Calculamos a distância de X para C, e elevamos ao quadrado
    return jnp.exp(-dist_sq / (2 * sigma ** 2))


def tanh_rbf(x, c, log_sigma):
    """
    Função de Base Radial Sigmoide Localizada (baseada em tanh).
    Phi(r) = 1 - tanh(beta * r^2), onde beta = 1 / (2 * sigma^2).

    Parâmetros:
    x (jnp.array): Vetor de entrada (e.g., [SOC]).
    c (jnp.array): Vetor do centro RBF.
    log_sigma (jnp.array): log do parâmetro de forma (largura), sigma.
    """
    sigma = jnp.exp(log_sigma) + 1e-6  # Garante sigma positivo

    # r^2 = Distância Euclidiana Quadrada
    dist_sq = jnp.sum((x - c) ** 2)

    # Beta = 1 / (2 * sigma^2)
    beta = 1.0 / (2.0 * sigma ** 2)

    # Phi(r) = 1 - tanh(beta * r^2)
    return 1.0 - jnp.tanh(beta * dist_sq)

def multiquadric_rbf(x, c, log_sigma):
    """
    Função de Base Radial Multiquadrática Inversa (IMQ).

    Parâmetros:
    x (jnp.array): Vetor de entrada (e.g., [SOC]).
    c (jnp.array): Vetor do centro RBF.
    log_sigma (jnp.array): log do parâmetro de forma (largura), sigma.
    """
    sigma = jnp.exp(log_sigma) + 1e-6

    # r^2 = Distância Euclidiana Quadrada
    dist_sq = jnp.sum((x - c)**2)

    # Phi(r) = sqrt(r^2 + sigma^2)
    return jnp.sqrt(dist_sq + sigma**2)


def inverse_multiquadric_rbf(x, c, log_sigma):
    """
    Função de Base Radial Multiquadrática Inversa (IMQ).

    Parâmetros:
    x (jnp.array): Vetor de entrada (e.g., [SOC]).
    c (jnp.array): Vetor do centro RBF.
    log_sigma (jnp.array): log do parâmetro de forma (largura), sigma.
    """
    sigma = jnp.exp(log_sigma) + 1e-6

    # r^2 = Distância Euclidiana Quadrada
    dist_sq = jnp.sum((x - c)**2)

    # Phi(r) = 1 / sqrt(r^2 + sigma^2)
    return 1.0 / jnp.sqrt(dist_sq + sigma**2)

def inverse_quadric_rbf(x, c, log_sigma):
    """
    Função de Base Radial Multiquadrática Inversa (IMQ).

    Parâmetros:
    x (jnp.array): Vetor de entrada (e.g., [SOC]).
    c (jnp.array): Vetor do centro RBF.
    log_sigma (jnp.array): log do parâmetro de forma (largura), sigma.
    """
    sigma = jnp.exp(log_sigma) + 1e-6

    # r^2 = Distância Euclidiana Quadrada
    dist_sq = jnp.sum((x - c)**2)

    # Phi(r) = 1 / (r^2 + sigma^2)
    return 1.0 / (dist_sq + sigma**2)


def thin_plate_spline_rbf(x, c, log_sigma):
    """
    Função de Base Radial Spline Fina (TPS).

    Parâmetros:
    x (jnp.array): Vetor de entrada (e.g., [SOC]).
    c (jnp.array): Vetor do centro RBF.
    """
    # r^2 = Distância Euclidiana Quadrada
    dist_sq = jnp.sum((x - c) ** 2)
    r = jnp.sqrt(dist_sq) + 1e-9  # Adiciona um epsilon para evitar log(0)

    # Phi(r) = r^2 * log(r)
    return dist_sq * jnp.log(r)

def predict_rbfn(params, inputs):
    """Forward pass da Rede Neural de Base Radial (RBFN)."""
    C, log_sigma, W_out, b_out = params

    # inputs: [input_size]
   # 1. Calcular a ativação (output da camada oculta RBF)
    # O mapeamento acontece sobre as linhas de C e log_sigma (eixo 0)
    # A função gaussian_rbf recebe um centro C_i (linha de C) e um sigma_i (elemento de log_sigma)
    rbf_outputs = jax.vmap(thin_plate_spline_rbf, in_axes=(None, 0, 0))(inputs, C, log_sigma)
    # rbf_outputs tem shape: [num_rbf_neurons]
   # 2. Camada de Saída Linear
    # logits = W_out * rbf_outputs + b_out
    # W_out: [output_size, num_rbf_neurons]
    # rbf_outputs: [num_rbf_neurons]
    logits = jnp.dot(W_out, rbf_outputs) + b_out

    # logits tem shape: [output_size] ->  [delta_R0, delta_R1, delta_C1]
    return logits

# --- New Hybrid ODE Model ---

def hybrid_battery_1rc_jax(t, x, args):
    params_nn, u_interp = args
    u = u_interp.evaluate(t)

    # The NN models the variation of model's parameters
    # Input -> SOC (x[0])
    # Outputs -> parameters variations (delta_R0, delta_R1, delta_C1)
    nn_input = jnp.array([x[0]])
    delta_R0, delta_R1, delta_C1 = predict_rbfn(params_nn, nn_input)
    R0 = 0.2462 * (1 + delta_R0)
    R1 = 2889.1884 * (1 + delta_R1)
    C1 = 3319.8907 * (1 + delta_C1)
    dx = [-0.3839 * u / 3440.05372, -1 / R1 / C1 * x[1] + 1 / C1 * u]
    dx = jnp.array(dx)
    return dx


term = ODETerm(hybrid_battery_1rc_jax)

t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))
u_interpolation = LinearInterpolation(ts=time, ys=u)

# --- NN and Optimization Configuration ---
input_size = 1# 1 inputs (SOC)
rbf_neurons = 32# 1 hidden layers of 64
output_size = 3# 3 outputs
solver = Dopri5()

# --- Create Initial Guess and Parameter Structures ---
key = random.key(0)
# Usa a nova função de inicialização
initial_params_rbfn = init_rbf_params(input_size, rbf_neurons, output_size, key)
# Armazena a estrutura dos parâmetros do RBFN para desachatamento posterior
flat_initial_nn_params, params_nn_struct = ravel_pytree(initial_params_rbfn)
len_nn_params = len(flat_initial_nn_params)

# Initial Guess for the state: [SOC, Vc]
x_initial_first_shot = jnp.array([0.98, 0.0])  # SOC em 98%, Vc em 0V
x_initial_shots_repeated = jnp.tile(x_initial_first_shot, n_shots)
initial_guess_np = np.concatenate([
    np.array(flat_initial_nn_params),
    x_initial_shots_repeated
])


# --- JIT-compiled Objective and Constraint Functions ---
# **CORRECTION**: The factory pattern is removed to avoid JIT closure issues.
# The unflattening logic is now explicitly inside each jitted function.

@jit
def objective_jax_nn(decision_vars):
    # Manually unflatten parameters inside the jitted function
    params_nn = params_nn_struct(decision_vars[:len_nn_params])
    x_initial_shots = decision_vars[len_nn_params:].reshape(n_shots, n_states)

    # args = (params_nn, u_interp)

    def simulate_shot(t_shot, w0):
        saveat = SaveAt(ts=t_shot)
        args = (params_nn, u_interpolation)
        sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=w0, saveat=saveat, args=args)
        return sol.ys

    # Simulation of the states prediction of each shot
    x_pred = jax.vmap(simulate_shot)(t_shots, x_initial_shots)

    def model_output_step(t, x_step, u_interp_obj):
        u = u_interp_obj.evaluate(t)

        p = jnp.array([1.02726610e+03,
                       -5.13266541e+03,
                       1.09109051e+04,
                       -1.28481333e+04,
                       9.13851696e+03,
                       -4.01608666e+03,
                       1.07265101e+03,
                       -1.65017255e+02,
                       1.36600705e+01,
                       3.10715139e+00])
        OCV = jnp.polyval(p, x_step[0])
        nn_input = jnp.array([x_step[0]])
        delta_R0, delta_R1, delta_C1 = predict_rbfn(params_nn, nn_input)
        R0 = 0.2462 * (1 + delta_R0)
        # A saída é: OCV + R0*u + Vc (x[1])
        y_pred_step = OCV + R0 * u + x_step[1]
        # y_pred_step = OCV + 0.2462 * u + x_step[1]
        return y_pred_step

    def process_shot_output(t_shot, x_shot, u_interp_obj):
        return jax.vmap(model_output_step, in_axes=(0, 0, None))(t_shot, x_shot, u_interp_obj)

    y_pred = jax.vmap(process_shot_output, in_axes=(0, 0, None))(t_shots, x_pred, u_interpolation)

    # Loss function
    return jnp.sum((y_pred - y_data) ** 2)


@jit
def continuity_constraints_jax_nn(decision_vars):
    params_nn_flat = decision_vars[:len_nn_params]
    params_nn = params_nn_struct(params_nn_flat)
    x_initial_shots = decision_vars[len_nn_params:].reshape(n_shots, n_states)
    args = (params_nn, u_interpolation)

    def get_end_state(t_shot, x0):
        sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=x0, args=args)
        return sol.ys[-1]

    x_end_of_shots = jax.vmap(get_end_state)(t_shots[:-1], x_initial_shots[:-1])
    # CORREÇÃO: X_end_of_shots -> x_end_of_shots
    return (x_end_of_shots - x_initial_shots[1:]).flatten()


# --- Create JIT-compiled Gradient and Jacobian Functions ---
objective_grad_func_nn = jit(jax.value_and_grad(objective_jax_nn))
constraints_jac_func_nn = jit(jax.jacrev(continuity_constraints_jax_nn))


# --- Wrapper Functions for SciPy Optimizer ---

def obj_for_scipy(dv_np):
    val, grad = objective_grad_func_nn(jnp.array(dv_np))
    return np.array(val), np.array(grad)


def cons_for_scipy(dv_np):
    return np.array(continuity_constraints_jax_nn(jnp.array(dv_np)))


def cons_jac_for_scipy(dv_np):
    jac_jax = constraints_jac_func_nn(jnp.array(dv_np))
    return np.array(jac_jax)  # SciPy can handle the jacobian structure directly


# --- Run Optimization ---

cons = ({'type': 'eq', 'fun': cons_for_scipy, 'jac': cons_jac_for_scipy})
max_iterations = 10000  # Increased iterations for the more complex model

with tqdm(total=max_iterations, desc="Optimizing Hybrid Model") as pbar:
    def callback(xk):
        pbar.update(1)


    print("\n--- Running Optimization with Neural Network ---")
    result = minimize(
        obj_for_scipy,
        initial_guess_np,
        method='SLSQP',
        jac=True,
        constraints=cons,
        options={'maxiter': max_iterations, 'disp': False},
        callback=callback
    )

print("\nOptimization finished with status:", result.message)

# --- Extract and Display Results ---

# Unflatten the final optimized parameters
params_nn_est = params_nn_struct(result.x[:len_nn_params])

# Use the first identified shot state as the initial state for the full simulation
x_initial_estimated = jnp.array(result.x[len_nn_params:len_nn_params + n_states])

print("\n--- Identification Results ---")

# --- Time-Domain Validation Plot ---
final_args = (params_nn_est, u_interpolation)
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=x_initial_estimated,
                        saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=100000)
yhat = final_sol.ys.flatten()
soc_series = final_sol.ys[:, 0]
soc_min = jnp.min(soc_series)
soc_max = jnp.max(soc_series)

def model_output_step(t, x_step, params_nn, u_interp_obj):
    u = u_interp_obj.evaluate(t)

    p = jnp.array([1.02726610e+03,
                   -5.13266541e+03,
                   1.09109051e+04,
                   -1.28481333e+04,
                   9.13851696e+03,
                   -4.01608666e+03,
                   1.07265101e+03,
                   -1.65017255e+02,
                   1.36600705e+01,
                   3.10715139e+00])
    OCV = jnp.polyval(p, x_step[0])
    nn_input = jnp.array([x_step[0]])
    delta_R0, delta_R1, delta_C1 = predict_rbfn(params_nn, nn_input)
    R0 = 0.2462 * (1 + delta_R0)
    # A saída é: OCV + R0*u + Vc (x[1])
    y_pred_step = OCV + R0 * u + x_step[1]
    # y_pred_step = OCV + 0.2462 * u + x_step[1]
    return y_pred_step


y_hat = jax.vmap(model_output_step, in_axes=(0, 0, None, None))(jnp.array(time), final_sol.ys, params_nn_est,
                                                                u_interpolation)
# Metrics
MSE = np.mean((y - y_hat) ** 2)
y_mean = jnp.mean(y)
RSS = jnp.sum((y - y_hat) ** 2)
TSS = jnp.sum((y - y_mean) ** 2)
r2 = 1.0 - (RSS / TSS)
print(f"R²: {r2:.4f}, MSE = {MSE:.4f}")

plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='Measured Data (y)', alpha=0.6)
plt.plot(time, y_hat, 'r--', label='Hybrid Model Prediction (y_hat)', linewidth=2)
plt.plot(time, y - y_hat, 'b-', label='Error', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (w)')
plt.title('Time-Domain Validation of the Hybrid Model')
plt.legend()
plt.grid(True)
plt.show()

# Loading validation dataset
DATA = loadmat('data_val.mat')
u = DATA['i']
y = DATA['v']
time = DATA['t']
time = time.reshape(-1)
u = u.reshape(-1)
y = y.reshape(-1)

# Signal generation parameters
N = time.shape[0]
Ts = time[1] - time[0]
fs = 1 / Ts
T = time[-1]  # Total time in seconds

print(f"\nValidation Dataset\nN ={N:.4f}\nfs={fs:.4f}\nT = {T:.4f}\nTs = {Ts:.4f}")

# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)

# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)
# Simulate the final model prediction
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=x_initial_estimated,
                        saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=100000)
# final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args_
yhat = final_sol.ys.flatten()

y_hat = jax.vmap(model_output_step, in_axes=(0, 0, None, None))(jnp.array(time), final_sol.ys, params_nn_est,
                                                                u_interpolation)
# # Plot final results
plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='True state', alpha=0.4)
plt.plot(time, y_hat, 'b--', label='Identified Model Prediction', linewidth=2)
plt.plot(time, y - y_hat, 'r', label='Residue', linewidth=2)
plt.xlabel('Time (s)')
plt.title('Model Identification Result')
plt.legend()
plt.grid(True)
plt.show()

# Metrics
MSE = np.mean((y - y_hat) ** 2)
y_mean = jnp.mean(y)
RSS = jnp.sum((y - y_hat) ** 2)
TSS = jnp.sum((y - y_mean) ** 2)
r2 = 1.0 - (RSS / TSS)
print(f"R²: {r2:.4f}, MSE = {MSE:.4f}")

soc_values_np = np.linspace(soc_min, soc_max, 100)
soc_values_jax = jnp.array(soc_values_np)

# 2. Definir a função que calcula as variações e os parâmetros
# Note que a função predict aceita um array 1D de [SOC] como input
def calculate_deltas(soc_scalar, params_nn):
    """Calcula as variações (deltas) para um SOC escalar."""
    nn_input = jnp.array([soc_scalar])
    delta_R0, delta_R1, delta_C1 = predict_rbfn(params_nn, nn_input)
    return delta_R0, delta_R1, delta_C1


# 3. Mapear a função para rodar em todos os valores de SOC (usando jax.vmap)
# in_axes=(0, None): Mapeia o primeiro argumento (soc_values_jax), mantém o segundo (params_nn_est)
vmap_calculate_deltas = vmap(calculate_deltas, in_axes=(0, None))
delta_R0_vec, delta_R1_vec, delta_C1_vec = vmap_calculate_deltas(soc_values_jax, params_nn_est)

# 4. Parâmetros Nominais (extraídos da sua função hybrid_battery_1rc_jax)
# R0 = 0.2462*(1+delta_R0)
# R1 = 2889.1884*(1+delta_R1)
# C1 = 3319.8907*(1+delta_C1)
R0_nominal = 0.2462
R1_nominal = 2889.1884
C1_nominal = 3319.8907

# 5. Calcular os parâmetros absolutos
R0_actual = R0_nominal * (1 + delta_R0_vec)
R1_actual = R1_nominal * (1 + delta_R1_vec)
C1_actual = C1_nominal * (1 + delta_C1_vec)

# 6. Plotar os resultados

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle('Variação dos Parâmetros do Modelo 1RC em função do SOC', fontsize=16)

# Plot R0
axs[0].plot(soc_values_np, R0_actual, 'r-', linewidth=2)
axs[0].axhline(R0_nominal, color='k', linestyle='--', alpha=0.6, label='Valor Nominal')
axs[0].set_ylabel('R_0 (\Omega)')
axs[0].set_title('Resistência Série (R_0)')
axs[0].grid(True)
axs[0].legend()

# Plot R1
axs[1].plot(soc_values_np, R1_actual, 'g-', linewidth=2)
axs[1].axhline(R1_nominal, color='k', linestyle='--', alpha=0.6, label='Valor Nominal')
axs[1].set_ylabel('R_1 (\Omega)')
axs[1].set_title('Resistência de Polarização (R_1)')
axs[1].grid(True)
axs[1].legend()

# Plot C1
axs[2].plot(soc_values_np, C1_actual, 'b-', linewidth=2)
axs[2].axhline(C1_nominal, color='k', linestyle='--', alpha=0.6, label='Valor Nominal')
axs[2].set_ylabel('$C_1$ (F)')
axs[2].set_xlabel('Estado de Carga (SOC)')
axs[2].set_title('Capacitância de Polarização ($C_1$)')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta para o suptitle
plt.show()