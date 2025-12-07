import pandas as pd
import matplotlib.pyplot as plt

df_err = pd.read_csv('/results/Erreur_TEST_MLP_LR0.0001_HL512-256-128.csv')
df_met = pd.read_csv('/results/Metrics_MLP_LR0.0001_HL512-256-128.csv')

# Utilisation de l'index comme axe temporel
t = df_err.index

# --- Graphique 1 : positions ---
plt.figure()
plt.plot(t, df_err['x_horizons_km'], label='x_horizons')
plt.plot(t, df_err['x_sgp4_km'], label='x_sgp4')
plt.plot(t, df_err['x_sgp4_km'] + df_err['err_x_pred'], label='x_sgp4 + err_x_pred')
plt.xlabel('Index (time step)')
plt.ylabel('X (km)')
plt.legend()
plt.title('Évolution X')
plt.savefig('/plots/graph_positions_LSTM.png')
plt.close()

# --- Graphique 2 : métriques ---
e = df_met.index
plt.figure()
plt.plot(e, df_met['train_rmse'], label='train_rmse')
plt.plot(e, df_met['valid_rmse'], label='val_rmse')
plt.plot(e, df_met['train_mse'], label='train_mse')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.title('Évolution des métriques')
plt.savefig('/plots/graph_metrics.png')
plt.close()
