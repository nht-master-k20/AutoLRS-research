import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_full_comparison():
    files = {
        "AutoLRS": "reproduce_vgg_log.csv",
        "Baseline": "baseline_vgg_log.csv",
        "Cosine": "cosine_vgg_log.csv"
    }

    dfs = {}
    for name, filepath in files.items():
        if os.path.exists(filepath):
            try:
                raw_df = pd.read_csv(filepath)
                # Chỉ lấy dòng có Accuracy (Val_Acc > 0)
                df = raw_df[pd.to_numeric(raw_df['Val_Acc'], errors='coerce') > 0].copy()
                # Tạo cột Epoch giả định dựa trên thứ tự xuất hiện nếu cần
                df['Relative_Epoch'] = range(1, len(df) + 1)
                dfs[name] = df
            except:
                pass

    if not dfs: return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    colors = {"AutoLRS": "#d62728", "Baseline": "gray", "Cosine": "green"}

    # --- 1. ACCURACY ---
    for name, df in dfs.items():
        sns.lineplot(data=df, x="Relative_Epoch", y="Val_Acc", label=name, ax=axs[0], color=colors[name], linewidth=2)
    axs[0].set_title("Top-1 Validation Accuracy", fontweight='bold')
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].set_xlabel("Epochs")
    axs[0].axhline(y=93.7, color='black', linestyle=':', label='Target 93.7%')
    axs[0].legend()

    # --- 2. LR SCHEDULE ---
    for name, df in dfs.items():
        # Vẽ theo Step cho chi tiết
        sns.lineplot(data=df, x="Step", y="LR", label=name, ax=axs[1], color=colors[name])
    axs[1].set_title("Learning Rate Schedule", fontweight='bold')
    axs[1].set_yscale("log")
    axs[1].set_ylabel("LR (Log Scale)")
    axs[1].set_xlabel("Training Steps")

    plt.tight_layout()
    plt.savefig("figure_1_reproduce.png", dpi=300)
    print("✅ Đã tạo Figure 1: figure_1_reproduce.png")
    plt.show()


if __name__ == "__main__":
    plot_full_comparison()