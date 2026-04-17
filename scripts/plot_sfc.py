import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_file(filepath, outdir):
    df = pd.read_csv(filepath)

    sessions = df.columns.tolist()
    n_sessions = len(sessions)
    n_trials = df.shape[0]

    # Generate unique colors for each session
    colors = plt.cm.hsv(np.linspace(0, 1, n_sessions + 1))[:-1]

    plt.figure(figsize=(20, 8))

    # Concatenate all column values in order: col0 trials, then col1 trials, etc.
    x_all = []
    y_all = []
    color_segments = []

    for col_idx, session in enumerate(sessions):
        for trial_idx in range(n_trials):
            x_all.append(col_idx * n_trials + trial_idx)
            y_all.append(df.iloc[trial_idx, col_idx])
            color_segments.append(colors[col_idx])

    # Plot as a continuous line with gradient coloring
    for i in range(len(x_all) - 1):
        plt.plot(x_all[i:i+2], y_all[i:i+2], color=color_segments[i], linewidth=1.5, alpha=0.8)

    # Add scatter points with session colors
    for col_idx, session in enumerate(sessions):
        x_vals = [col_idx * n_trials + i for i in range(n_trials)]
        y_vals = df.iloc[:, col_idx].values
        plt.scatter(x_vals, y_vals, c=[colors[col_idx]], s=40, label=session, zorder=5)

    # Set x-ticks at the start of each column
    tick_positions = [col_idx * n_trials for col_idx in range(n_sessions)]
    plt.xticks(tick_positions, sessions, rotation=45)

    plt.title(f"SFC Across Sessions (Time Progresses Down Columns)\n{os.path.basename(filepath)}")
    plt.xlabel("Session")
    plt.ylabel("SFC value")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    out_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(outdir, f"{out_name}_sfc_trials.png") #Currently saves in the same folder as the CSV, change if you want to save elsewhere

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


def run(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))

    if not files:
        print("No CSV files found.")
        return

    for f in files:
        plot_file(f, folder)


if __name__ == "__main__":
    folder = r"" #CSV Containing folder
    run(folder)