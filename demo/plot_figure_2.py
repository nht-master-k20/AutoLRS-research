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

    # Smooth window: 50 bước
    SMOOTH = 50
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(10, 7))
    colors = {"AutoLRS": "#d62728", "Baseline": "gray", "Cosine": "green"}

    for name, filepath in files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Lọc dòng có Train_Loss
                df = df[pd.to_numeric(df['Train_Loss'], errors='coerce').notnull()].copy()
                df['Train_Loss'] = df['Train_Loss'].astype(float)

                # Làm mượt
                df['Loss_Smooth'] = df['Train_Loss'].rolling(window=SMOOTH).mean()

                # Vẽ 50k bước đầu tiên (để giống Figure 2b) hoặc toàn bộ
                sns.lineplot(data=df, x="Step", y="Loss_Smooth", label=name, color=colors[name], linewidth=1.5)
            except:
                pass

    plt.title("Figure 2 Reproduction: Training Loss", fontweight='bold')
    plt.ylabel("Training Loss")
    plt.xlabel("Training Steps")
    plt.xlim(0, 50000)  # Zoom vào giai đoạn đầu như paper
    plt.legend()

    plt.savefig("figure_2_reproduce.png", dpi=300)
    print("✅ Đã tạo Figure 2: figure_2_reproduce.png")
    plt.show()


if __name__ == "__main__":
    plot_figure_2()