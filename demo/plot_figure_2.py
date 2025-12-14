import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_figure_2():
    files = {
        "AutoLRS": "reproduce_vgg_log.csv",
        "Baseline": "baseline_vgg_log.csv",
        "Cosine": "cosine_vgg_log.csv"
    }
    colors = {"AutoLRS": "#d62728", "Baseline": "#7f7f7f", "Cosine": "#2ca02c"}

    # Tăng độ làm mượt để hình đẹp hơn
    SMOOTH_WINDOW = 100

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.figure(figsize=(12, 8))

    for name, filepath in files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Lọc dữ liệu Training Loss (bỏ các dòng trống)
                df = df[pd.to_numeric(df['Train_Loss'], errors='coerce').notnull()].copy()
                df['Train_Loss'] = df['Train_Loss'].astype(float)

                # Kỹ thuật Rolling Mean để làm mượt
                df['Loss_Smooth'] = df['Train_Loss'].rolling(window=SMOOTH_WINDOW).mean()

                # Vẽ biểu đồ
                sns.lineplot(data=df, x="Step", y="Loss_Smooth", label=name, color=colors[name], linewidth=2)
            except Exception as e:
                print(f"Lỗi file {name}: {e}")

    plt.title("Training Loss Convergence", fontweight='bold', fontsize=18)
    plt.ylabel("Training Loss (Smoothed)", fontsize=14)
    plt.xlabel("Training Steps", fontsize=14)

    # Giới hạn trục Y để tập trung vào vùng hội tụ (Bỏ qua đoạn loss nổ ban đầu)
    plt.ylim(0, 1.5)

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("figure_2_reproduce.png", dpi=300)
    print("✅ Đã tạo Figure 2: figure_2_reproduce.png")


if __name__ == "__main__": plot_figure_2()