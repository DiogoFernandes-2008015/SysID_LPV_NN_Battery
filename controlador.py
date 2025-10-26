import subprocess
import sys

# --- Defina os experimentos MLP ---
experimentos = [
    #{'activation': 'relu', 'neurons': 16,'decimate': 1,'n_shots': 10},
    #{'activation': 'tanh', 'neurons': 16,'decimate': 1,'n_shots': 10},
    #{'activation': 'relu', 'neurons': 32, 'decimate': 1,'n_shots': 10},
    {'activation': 'tanh', 'neurons': 32, 'decimate': 1,'n_shots': 10},
    {'activation': 'relu', 'neurons': 64, 'decimate': 1,'n_shots': 10},
    {'activation': 'tanh', 'neurons': 64, 'decimate': 1,'n_shots': 10}
]
# --- Defina os experimentos RBF ---
#experimentos = [
  #  {'activation': 'tanh', 'neurons': 64, 'decimate': 10,'n_shots': 10},
  #  {'activation': 'gaussian', 'neurons': 64, 'decimate': 2,'n_shots': 50},
  #  {'activation': 'iq', 'neurons': 32, 'decimate': 2,'n_shots': 10},
  #  {'activation': 'iq', 'neurons': 64, 'decimate': 2,'n_shots': 10},
  #  {'activation': 'imq', 'neurons': 32, 'decimate': 2,'n_shots': 10},
  #  {'activation': 'imq', 'neurons': 64, 'decimate': 2,'n_shots': 10},
  #  {'activation': 'tanh', 'neurons': 32, 'decimate': 2,'n_shots': 50},
  #  {'activation': 'tanh', 'neurons': 64, 'decimate': 2,'n_shots': 50},
  #  {'activation': 'thps', 'neurons': 32, 'decimate': 1,'n_shots': 10},
  #  {'activation': 'thps', 'neurons': 64, 'decimate': 1,'n_shots': 10},
#]
# Nome do seu script principal
script_principal = 'ID_Hyb_Grey_JAX_DNN_1RC.py'  # <-- Certifique-se que o nome está correto
#script_principal = 'ID_Hyb_Grey_JAX_RBF_1RC.py'  # <-- Certifique-se que o nome está correto

# Pega o caminho do executável do Python (garante que use o mesmo 'python')
python_executable = sys.executable

print(f"--- Iniciando Bateria de Experimentos ---")

for i, config in enumerate(experimentos):
    act = config['activation']
    neu = str(config['neurons'])
    dec = str(config['decimate'])
    ns = str(config['n_shots'])# Argumentos de linha de comando são strings

    print(f"\n[Experimento {i + 1}/{len(experimentos)}]: Ativação={act}, Neurônios={neu}, Decimação={dec}, N_shots={ns}")

    # Constrói o comando
    command = [
        python_executable,
        script_principal,
        '--activation', act,
        '--neurons', neu,
        '--decimate', dec,
        '--n_shots', ns,
    ]

    # Executa o comando e espera ele terminar
    try:
        # check=True faz o script parar se o treinamento.py der erro
        subprocess.run(command, check=True)
        print(f"[Experimento {i + 1}] Concluído com sucesso.")

    except subprocess.CalledProcessError as e:
        print(f"!!! [Experimento {i + 1}] FALHOU com erro: {e} !!!")

print("\n--- Todos os experimentos foram concluídos. ---")