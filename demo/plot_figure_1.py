import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CẤU HÌNH ---
FILES = {
    "AutoLRS": "reproduce_vgg_log.csv",
    "Baseline": "baseline_vgg_log.csv",
    "Cosine": "cosine_vgg_log.csv"
}
COLORS = {"AutoLRS": "#d62728", "Baseline": "#7f7f7f", "Cosine": "#2ca02c"}

# Số lượng Epoch để tính trung bình trượt
SMOOTH_WINDOW = 40


def load_and_process_data():
    """Đọc và tiền xử lý dữ liệu."""
    dfs = {}
    for name, filepath in FILES.items():
        if os.path.exists(filepath):
            try:
                raw_df = pd.read_csv(filepath)
                df = raw_df[pd.to_numeric(raw_df['Val_Acc'], errors='coerce') > 0].copy()
                df['Epoch'] = pd.to_numeric(df['Epoch'])
                if 'LR' in df.columns:
                    df['LR'] = pd.to_numeric(df['LR'], errors='coerce')
                dfs[name] = df
            except Exception as e:
                print(f"Skipping {name}: {e}")
    return dfs if dfs else None


def plot_accuracy(dfs):
    """Vẽ biểu đồ Accuracy với kỹ thuật Smoothing."""
    if not dfs: return

    plt.figure(figsize=(10, 7))

    for name, df in dfs.items():
        # Tính toán cột dữ liệu đã làm mượt (Moving Average)
        df['Val_Acc_Smooth'] = df['Val_Acc'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

        # Vẽ đường MƯỢT
        sns.lineplot(
            data=df, x="Epoch", y="Val_Acc_Smooth",
            label=f"{name}",
            color=COLORS[name], linewidth=2.5
        )

    plt.title(f"Top-1 Validation Accuracy", fontweight='bold', fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    output_file = "plot_accuracy.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã tạo biểu đồ Accuracy (Smooth): {output_file}")


def plot_learning_rate(dfs):
    """Vẽ biểu đồ Learning Rate CÓ Smoothing."""
    if not dfs: return

    plt.figure(figsize=(10, 7))
    for name, df in dfs.items():
        # --- CẬP NHẬT: Thêm logic Smooth cho LR ---
        # AutoLRS có LR dao động mạnh, làm mượt sẽ dễ nhìn xu hướng hơn
        df['LR_Smooth'] = df['LR'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

        # Vẽ đường đã làm mượt
        sns.lineplot(
            data=df, x="Epoch", y="LR_Smooth",
            label=name,
            color=COLORS[name], linewidth=2.5
        )

    plt.title("Learning Rate Schedule", fontweight='bold', fontsize=16)
    plt.yscale("log") # Vẫn giữ thang đo Logarit để nhìn rõ độ lớn
    plt.ylabel("Learning Rate", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)

    output_file = "plot_learning_rate.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã tạo biểu đồ Learning Rate (Smooth): {output_file}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    dataframes = load_and_process_data()
    if dataframes:
        plot_accuracy(dataframes)
        plot_learning_rate(dataframes)