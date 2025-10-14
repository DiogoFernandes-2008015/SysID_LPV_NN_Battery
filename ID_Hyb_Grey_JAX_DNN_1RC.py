import numpy as np
import scipy as sc
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat
from tqdm import tqdm # For the progress bar
import os
import scipy.io
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, LinearInterpolation
from jax import random, grad, jit, vmap
from jax.flatten_util import ravel_pytree
    
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

fig, axs = plt.subplots(2, 1, sharex=True) # sharex makes sense for time series

# Plot 1: Input u
axs[0].plot(time, u, color='b') # Added color for clarity
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

plt.tight_layout() # Adjusts subplot params for a tight layout
plt.show()

if time is not None:
    time = time.flatten()
    u = u.flatten()
    y = y.flatten()

    N = time.shape[0]
    Ts = time[1] - time[0]
    fs = 1 / Ts
    T = time[-1]
    print(f"N={N}, fs={fs}, T={T}, Ts={Ts}")

# --- Neural Network Helper Functions (from JAX ANN example) ---

def random_layer_params(m, n, key, scale=1e-2):
  """A helper function to randomly initialize weights and biases for a dense layer."""
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
  """Initialize all layers for a fully-connected neural network with sizes "sizes"."""
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# def relu(x):
#   return jnp.maximum(0, x)

def predict(params, inputs):
  """Neural network forward pass."""
  activations = inputs
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = jnp.tanh(outputs) # Changed from relu to tanh
    # activations = relu(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits

# --- New Hybrid ODE Model ---

def hybrid_battery_1rc_jax(t, x, args):
    params_nn, u_interp = args
    u = u_interp.evaluate(t)
    
    # The NN models the variation of model's parameters
    # Input -> SOC (x[0])
    # Outputs -> parameters variations (delta_R0, delta_R1, delta_C1)
    nn_input = jnp.array([x[0]])
    delta_R0, delta_R1, delta_C1 = predict(params_nn, nn_input)
    R0 = 0.2462*(1+delta_R0)
    R1 = 2889.1884*(1+delta_R1)
    C1 = 3319.8907*(1+delta_C1)
    dx  = [-0.3839*u/3440.05372, -1/R1/C1*x[1]+1/C1*u]
    dx = jnp.array(dx)
    return dx

term = ODETerm(hybrid_battery_1rc_jax)

# --- Setup for Multiple Shooting ---
n_shots = 100 # random
n_timesteps_per_shot = N // n_shots

t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))
u_interpolation = LinearInterpolation(ts=time, ys=u)

# --- NN and Optimization Configuration ---
nn_layer_sizes = [1, 64, 3]  # 2 inputs (w, u), 2 hidden layers of 10, 1 output
solver = Dopri5()

# --- Create Initial Guess and Parameter Structures ---
key = random.key(0)
initial_params_nn = init_network_params(nn_layer_sizes, key)
# Store the structure of the NN parameters for later unflattening
flat_initial_nn_params, params_nn_struct = ravel_pytree(initial_params_nn)
len_nn_params = len(flat_initial_nn_params)

# 2. Palpite para o PRIMEIRO Estado Inicial (x_0,1)
# Estado inicial: [SOC, Vc]
x_initial_first_shot = jnp.array([0.98, 0.0]) # SOC em 98%, Vc em 0V

# 3. Palpite para os Estados dos Shots Intermediários
x_initial_shots_repeated = jnp.tile(x_initial_first_shot, n_shots) 


# Create the full, flattened initial guess vector for the optimizer
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
    x_initial_shots = decision_vars[len_nn_params:].reshape(n_shots, 2) 
    #args = (params_nn, u_interp)

    def simulate_shot(t_shot, w0):
        saveat = SaveAt(ts=t_shot)
        args = (params_nn, u_interpolation)
        sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=w0, saveat=saveat, args=args)
        return sol.ys

    # Simulation of the states prediction of each shot
    x_pred = jax.vmap(simulate_shot)(t_shots, x_initial_shots)
    
    # Função que calcula a saída para UM PASSO de tempo
    def model_output_step(t, x_step, u_interp_obj):
        # Obtemos a corrente (u) usando o objeto interpolador
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
                       3.10715139e+00]) # Polinômio
        OCV = jnp.polyval(p, x_step[0])
        # A saída é: OCV + R0*u + Vc (x[1])
        y_pred_step = OCV + 0.2462 * u + x_step[1]
        return y_pred_step
    
    def process_shot_output(t_shot, x_shot, u_interp_obj):
        # Mapeia sobre o tempo (N_steps). t, x_step variam (0), R0 e u_interp são constantes (None).
        return jax.vmap(model_output_step, in_axes=(0, 0, None))(t_shot, x_shot, u_interp_obj)
    
    # Mapeia sobre os shots (N_shots). t_shot e x_shot variam (0), R0 e u_interp são constantes (None).
    y_pred = jax.vmap(process_shot_output, in_axes=(0, 0, None))(t_shots, x_pred, u_interpolation)
    
    
    return jnp.sum((y_pred - y_data)**2)

@jit
def continuity_constraints_jax_nn(decision_vars):
    params_nn_flat = decision_vars[:len_nn_params]
    params_nn = params_nn_struct(params_nn_flat)
    x_initial_shots = decision_vars[len_nn_params:].reshape(n_shots, 2) 
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
    return np.array(jac_jax) # SciPy can handle the jacobian structure directly

# --- Run Optimization ---

cons = ({'type': 'eq', 'fun': cons_for_scipy, 'jac': cons_jac_for_scipy})
max_iterations = 10000 # Increased iterations for the more complex model

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
theta1_est = result.x[0]
theta3_est = result.x[1]
params_nn_est = params_nn_struct(result.x[2:len_nn_params+2])

# Use the first identified shot state as the initial state for the full simulation
w0_est = result.x[len_nn_params+2]

print("\n--- Identification Results ---")
print(f"Estimated physical parameters: theta1 = {theta1_est:.4f}, theta3 = {theta3_est:.4f}")

# --- Time-Domain Validation Plot ---
final_args = (theta1_est, theta3_est, params_nn_est, u_interpolation)
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=w0_est,
                        saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=16384)
yhat = final_sol.ys.flatten()
y_hat_train = yhat
x_train = u

plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='Measured Data (y)', alpha=0.6)
plt.plot(time, yhat, 'r--', label='Hybrid Model Prediction (y_hat)', linewidth=2)
plt.plot(time, y - yhat, 'b-', label='Error', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (w)')
plt.title('Time-Domain Validation of the Hybrid Model')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(y,yhat, 'ko')
plt.xlabel('Measured (y)')
plt.ylabel('Predicted (yhat)')
plt.grid(True)
plt.show()

DATA = loadmat('pythondataval.mat')
u = DATA['u']
y = DATA['y']/10
time = DATA['t']
time = time.reshape(-1)
u = u.reshape(-1)
y = y.reshape(-1)

# Signal generation parameters
# N = 2048  # Number of samples (power of 2 is efficient for FFT)
N = time.shape[0]
Ts = time[1]-time[0]
fs = 1/Ts
T = time[-1]  # Total time in seconds

print(N, fs, T, Ts)

n_shots = 43 # 8150 / 163 = 50 data points per shot.
n_timesteps_per_shot = N // n_shots

# Reshape data into batches for multiple shooting
t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))

# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)
# Simulate the final model prediction
# --- Time-Domain Validation Plot ---
final_args = (theta1_est, theta3_est, params_nn_est, u_interpolation)
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=w0_est,
                        saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=16384)
yhat = final_sol.ys.flatten()

plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='Measured Data (y)', alpha=0.6)
plt.plot(time, yhat, 'r--', label='Hybrid Model Prediction (y_hat)', linewidth=2)
plt.plot(time, y - yhat, 'b-', label='Error', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (w)')
plt.title('Time-Domain Validation of the Hybrid Model')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(y,yhat, 'ko')
plt.xlabel('Measured (y)')
plt.ylabel('Predicted (yhat)')
plt.grid(True)
plt.show()
y_hat_val = yhat
x_valid = u

results = {'x_valid':x_valid,'x_train':x_train,'y_hat_train':y_hat_train,'y_hat_val':y_hat_val}


sc.io.savemat('resultados_hib_nn.mat',results)