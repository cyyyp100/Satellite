import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
mlp = pd.read_csv('/Users/cyprienvial/Documents/3A/Satellite/results/Erreur_TEST_MLP_LR0.0001_HL512-256-128.csv')
lstm = pd.read_csv('/Users/cyprienvial/Documents/3A/Satellite/results/Test_Predictions_LSTM_H128_L3.csv')
trans = pd.read_csv('/Users/cyprienvial/Documents/3A/Satellite/results/Test_Predictions.csv')

# Restrict to first 500 steps
N = 500
mlp = mlp.iloc[:N]
lstm = lstm.iloc[:N]
trans = trans.iloc[:N]

# ----------- MLP FIXED VERSION -----------
plt.figure()
plt.plot(mlp.index, mlp['x_horizons_km'], label='x_horizon', linestyle='--')
plt.plot(mlp.index, mlp['x_sgp4_km'], label='x_sgp4', linestyle='-.')
plt.plot(mlp.index, mlp['x_sgp4_km'] + mlp['err_x_pred'], label='x_sgp4 + pred_err', linewidth=2)
plt.xlabel("time step")
plt.ylabel("x (km)")
plt.legend()
plt.title("MLP – X evolution (500 steps)")
plt.savefig("/Users/cyprienvial/Documents/3A/Satellite/plots/graph_mlp_500_fixed.png")
plt.show()

# ----------- LSTM FIXED VERSION -----------
plt.figure()
plt.plot(lstm.index, lstm['x_horizons_km'], label='x_horizon', linestyle='--')
plt.plot(lstm.index, lstm['x_sgp4_km'], label='x_sgp4', linestyle='-.')
plt.plot(lstm.index, lstm['x_corrected'], label='x_corrected', linewidth=2)
plt.xlabel("time step")
plt.ylabel("x (km)")
plt.legend()
plt.title("LSTM – X evolution (500 steps)")
plt.savefig("/Users/cyprienvial/Documents/3A/Satellite/plots/graph_lstm_500_fixed.png")
plt.show()

# ----------- TRANSFORMER FIXED VERSION -----------
plt.figure()
plt.plot(trans.index, trans['x_horizons_km'], label='x_horizon', linestyle='--')
plt.plot(trans.index, trans['x_sgp4_km'], label='x_sgp4', linestyle='-.')
plt.plot(trans.index, trans['x_corrected'], label='x_corrected', linewidth=2)
plt.xlabel("time step")
plt.ylabel("x (km)")
plt.legend()
plt.title("Transformer – X evolution (500 steps)")
plt.savefig("/Users/cyprienvial/Documents/3A/Satellite/plots/graph_transformer_500_fixed.png")
plt.show()
