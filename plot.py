import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FILES_TO_COMPARE = {
    'No Compensation': 'build/traj_no_comp.csv',      
    'DOB-MPC (Compensated)': 'build/traj_with_comp.csv' 
}

# --- 2. 绘图样式配置 ---
COLORS = {'No Compensation': 'gray', 'DOB-MPC (Compensated)': 'red'}
STYLES = {'No Compensation': '--', 'DOB-MPC (Compensated)': '-'}
LINEWIDTHS = {'No Compensation': 2, 'DOB-MPC (Compensated)': 2}

def calculate_metrics(df):

    error_sq = (df['x'] - df['x_ref'])**2 + (df['y'] - df['y_ref'])**2
    rmse_total = np.sqrt(np.mean(error_sq))

    lateral_error = df['y'] - df['y_ref']
    rmse_lat = np.sqrt(np.mean(lateral_error**2))
    
    return rmse_total, rmse_lat, lateral_error

def plot_analysis():
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')

    data_store = {}
    rmse_results = {}

    print(f"{'Experiment':<25} | {'Total RMSE (m)':<15} | {'Lateral RMSE (m)':<15}")
    print("-" * 60)

    for label, filepath in FILES_TO_COMPARE.items():
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
            
        df = pd.read_csv(filepath)
        rmse_tot, rmse_lat, lat_err_series = calculate_metrics(df)
        

        df['lat_error'] = lat_err_series
        data_store[label] = df
        rmse_results[label] = rmse_tot
        
        print(f"{label:<25} | {rmse_tot:.4f} m{'':<8} | {rmse_lat:.4f} m")

    if not data_store:
        print("No data loaded!")
        return


    plt.figure(figsize=(10, 8))

    first_key = list(data_store.keys())[0]
    ref_df = data_store[first_key]
    plt.plot(ref_df['x_ref'], ref_df['y_ref'], 'k--', linewidth=2, alpha=0.6, label='Reference Path')

    for label, df in data_store.items():
        rmse_val = rmse_results[label]
        label_text = f"{label} (RMSE: {rmse_val:.3f}m)"
        
        plt.plot(df['x'], df['y'], 
                 color=COLORS[label], 
                 linestyle=STYLES[label], 
                 linewidth=LINEWIDTHS[label],
                 label=label_text)

    plt.title('Vehicle Trajectory Comparison', fontsize=16)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.axis('equal') 
    plt.grid(True)

    plt.figure(figsize=(12, 6))
    for label, df in data_store.items():
        plt.plot(df['t'], df['lat_error'], 
                 color=COLORS[label], 
                 linestyle=STYLES[label], 
                 linewidth=2,
                 label=label)
    
    plt.title('Lateral Tracking Error over Time', fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Lateral Error (y - y_ref) [m]', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.figure(figsize=(12, 6))

    first_df = list(data_store.values())[0]
    if 'real_dist_y' in first_df.columns:
        plt.plot(first_df['t'], first_df['real_dist_y'], 'k', linewidth=2, alpha=0.3, label='Real Disturbance (Truth)')

    for label, df in data_store.items():
        if 'Compensated' in label and 'est_dist_y' in df.columns:
            plt.plot(df['t'], df['est_dist_y'], color='red', linewidth=1.5, label=f'Estimated ({label})')

    plt.title('Disturbance Observer Performance', fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Disturbance Force', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    plot_analysis()