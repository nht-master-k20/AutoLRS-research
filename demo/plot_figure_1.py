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
SMOOTH_WINDOW = 5


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
    if not dfs: return

    plt.figure(figsize=(16, 9))

    threshold = 85

    # Vẽ đường kẻ ngang mốc 85%
    plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.text(0, threshold + 0.5, f'TARGET: {threshold}%', color='black', fontsize=12, fontweight='bold')

    for name, df in dfs.items():
        # Làm mượt đường vẽ để biểu đồ đẹp hơn
        df['Val_Acc_Smooth'] = df['Val_Acc'].rolling(window=5, min_periods=1).mean()

        # Vẽ đường biểu đồ chính
        sns.lineplot(
            data=df, x="Epoch", y="Val_Acc_Smooth",
            label=name,
            color=COLORS[name], linewidth=2
        )

        # Tìm điểm đầu tiên chạm mốc 85% (sử dụng dữ liệu gốc - raw data để chính xác)
        hit_row = df[df['Val_Acc'] >= threshold].sort_values('Epoch').head(1)

        if not hit_row.empty:
            epoch = hit_row['Epoch'].values[0]
            # Vẽ điểm đánh dấu (Marker)
            plt.scatter(epoch, threshold, color=COLORS[name], s=150, zorder=10, edgecolors='white', linewidth=2)

            # Thêm chú thích text
            plt.annotate(
                f'{name}\nEpoch {epoch}',
                xy=(epoch, threshold),
                xytext=(10, -30) if name != "Baseline" else (10, -40),  # Chỉnh vị trí text tránh đè nhau
                textcoords='offset points',
                arrowprops=dict(facecolor=COLORS[name], arrowstyle="->", alpha=0.8),
                fontsize=11, fontweight='bold', color=COLORS[name],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS[name], alpha=0.9)
            )

    plt.title(f"Top-1 Validation Accuracy (threshold={threshold}%)", fontweight='bold', fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)

    plt.xlim(0, 200)

    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("plot_accuracy.png", dpi=300)
    print("✅ Đã tạo biểu đồ highlight mốc 85%: plot_accuracy_highlight_85.png")


def plot_learning_rate(dfs):
    """Vẽ biểu đồ Learning Rate CÓ Smoothing và Highlight."""
    if not dfs: return

    plt.figure(figsize=(16, 9))
    for name, df in dfs.items():
        # AutoLRS có LR dao động mạnh, làm mượt sẽ dễ nhìn xu hướng hơn
        df['LR_Smooth'] = df['LR'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

        sns.lineplot(
            data=df, x="Epoch", y="LR_Smooth",
            label=name,
            color=COLORS[name], linewidth=2.5
        )

    plt.title("Learning Rate Schedule", fontweight='bold', fontsize=18)
    plt.yscale("log")
    plt.ylabel("Learning Rate", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)

    plt.xlim(0, 200)
    plt.ylim(1e-3, 0.2)

    # --- PHẦN HIGHLIGHT QUAN TRỌNG ---

    # 1. Highlight pha Warm-up của AutoLRS (Đường đỏ vọt lên)
    # Tọa độ xy là điểm mũi tên chỉ vào, xytext là vị trí đặt chữ
    plt.annotate('Auto Warm-up\n(Exploration)',
                 xy=(15, 0.08), xytext=(40, 0.12),
                 arrowprops=dict(facecolor=COLORS['AutoLRS'], shrink=0.05),
                 fontsize=12, fontweight='bold', color=COLORS['AutoLRS'],
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS['AutoLRS'], alpha=0.8))

    # 2. Highlight pha giảm LR thủ công của Baseline (Đường xám gãy khúc)
    # Thường Step Decay giảm ở epoch 150
    plt.annotate('Manual Step Decay\n(Exploitation)',
                 xy=(150, 0.1), xytext=(110, 0.005),
                 arrowprops=dict(facecolor=COLORS['Baseline'], shrink=0.05),
                 fontsize=12, fontweight='bold', color=COLORS['Baseline'],
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS['Baseline'], alpha=0.8))

    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)

    output_file = "plot_learning_rate.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã tạo biểu đồ Learning Rate (Annotated): {output_file}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    dataframes = load_and_process_data()
    if dataframes:
        plot_accuracy(dataframes)
        plot_learning_rate(dataframes)