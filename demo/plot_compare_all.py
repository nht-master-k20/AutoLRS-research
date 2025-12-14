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

    # Màu sắc chuẩn Paper: AutoLRS đỏ, Baseline xám/đen, Cosine xanh lá
    colors = {"AutoLRS": "#d62728", "Baseline": "#7f7f7f", "Cosine": "#2ca02c"}

    dfs = {}
    for name, filepath in files.items():
        if os.path.exists(filepath):
            try:
                raw_df = pd.read_csv(filepath)
                # Chỉ lấy các dòng có Validation Accuracy
                df = raw_df[pd.to_numeric(raw_df['Val_Acc'], errors='coerce') > 0].copy()
                df['Epoch'] = pd.to_numeric(df['Epoch'])
                dfs[name] = df
            except Exception as e:
                print(f"Skipping {name}: {e}")

    if not dfs:
        print("Không tìm thấy dữ liệu log!")
        return

    # Cấu hình giao diện đẹp
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))

    # --- BIỂU ĐỒ 1: ACCURACY ---
    for name, df in dfs.items():
        sns.lineplot(data=df, x="Epoch", y="Val_Acc", label=name, ax=axs[0], color=colors[name], linewidth=2.5)

    axs[0].set_title("Top-1 Validation Accuracy", fontweight='bold', fontsize=16)
    axs[0].set_ylabel("Accuracy (%)", fontsize=14)
    axs[0].set_xlabel("Epochs", fontsize=14)
    axs[0].legend(loc="lower right")
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # --- BIỂU ĐỒ 2: LEARNING RATE ---
    for name, df in dfs.items():
        sns.lineplot(data=df, x="Step", y="LR", label=name, ax=axs[1], color=colors[name], linewidth=2)

    axs[1].set_title("Learning Rate Schedule", fontweight='bold', fontsize=16)
    axs[1].set_yscale("log")  # LR dùng thang đo Logarit mới nhìn rõ
    axs[1].set_ylabel("Learning Rate (Log Scale)", fontsize=14)
    axs[1].set_xlabel("Training Steps", fontsize=14)
    axs[1].grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figure_1_reproduce.png", dpi=300)
    print("✅ Đã tạo Figure 1: figure_1_reproduce.png")
    # plt.show() # Bỏ comment nếu chạy trên máy có màn hình


if __name__ == "__main__": plot_full_comparison()