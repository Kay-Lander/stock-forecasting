import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)


def compute_metrics(y_true, y_pred, model_name = 'Model', ticker=''):
    '''
    Compute RMSE, MAE, MAPE, and Directional Accuracy.
    Returns dict of metrics and prints a summary.
    '''

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) **2))
    mae = np.mean(np.abs(y_true -y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Directional accuracy: did we predict up/down correctly?
    actual_dir    = np.diff(y_true) > 0
    predicted_dir = np.diff(y_pred) > 0
    dir_accuracy  = np.mean(actual_dir == predicted_dir) * 100

    metrics = {
        'model': model_name,
        'ticker': ticker,
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'mape': round(mape, 2),
        'dir_acc': round(dir_accuracy, 1)
    }

    print(f'\n{'='*50}')
    print(f'  {model_name} — {ticker}')
    print(f'{'='*50}')
    print(f'  RMSE               : {rmse:.4f}')
    print(f'  MAE                : {mae:.4f}')
    print(f'  MAPE               : {mape:.2f}%')
    print(f'  Directional Acc.   : {dir_accuracy:.1f}%')
 
    return metrics


def plot_predictions(dates, y_true, y_pred, model_name, ticker):
    '''Plot actual vs predicted prices and save to outputs/figures.'''
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_true, label='Actual',    color='#1B3A6B', linewidth=2)
    plt.plot(dates, y_pred, label='Predicted', color='#C45C00', linewidth=1.5, linestyle='--')
    plt.title(f'{model_name} — {ticker} Price Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    fname = f'outputs/figures/{ticker}_{model_name.replace(" ","_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Chart saved to {fname}')
 
 
def save_results(metrics_list):
    '''Save all model results to a CSV for the README table.'''
    df = pd.DataFrame(metrics_list)
    df.to_csv('outputs/results/model_comparison.csv', index=False)
    print('\n===== FULL RESULTS TABLE =====')
    print(df.to_string(index=False))
    return df


