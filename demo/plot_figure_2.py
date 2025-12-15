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
    SMOOTH_WINDOW = 1000

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(16, 9))

    max_step = 0  # Biến để tìm step lớn nhất

    for name, filepath in files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df = df[pd.to_numeric(df['Train_Loss'], errors='coerce').notnull()].copy()
                df['Train_Loss'] = df['Train_Loss'].astype(float)
                df['Step'] = pd.to_numeric(df['Step'], errors='coerce')

                # Cập nhật max step để set limit
                if not df.empty:
                    current_max = df['Step'].max()
                    if current_max > max_step: max_step = current_max

                # Rolling Mean
                df['Loss_Smooth'] = df['Train_Loss'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

                sns.lineplot(data=df, x="Step", y="Loss_Smooth", label=name, color=colors[name], linewidth=2.5)
            except Exception as e:
                print(f"Lỗi file {name}: {e}")

    plt.title("Training Loss Convergence", fontweight='bold', fontsize=18)
    plt.ylabel("Training Loss", fontsize=14)
    plt.xlabel("Training Steps", fontsize=14)

    # --- CẤU HÌNH KHUNG HÌNH (QUAN TRỌNG ĐỂ THẤY MŨI TÊN) ---
    plt.ylim(0, 2)  # Mở rộng trục Y lên 1.5 để mũi tên không bị cắt
    plt.xlim(left=0)  # Bắt buộc trục X bắt đầu từ 0

    # --- HIGHLIGHT 1: FAST CONVERGENCE (ĐOẠN ĐẦU) ---
    # Mũi tên chỉ vào Step 5000, Loss ~1.0
    plt.annotate('Fast Drop\n(Speed)',
                 xy=(20000, 1.2),  # Điểm mũi tên chỉ vào (Thấp xuống xíu cho dễ trúng đường đỏ)
                 xytext=(30000, 1.5),  # Vị trí đặt chữ
                 arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2),
                 fontsize=12, fontweight='bold', color='#d62728',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9))

    # --- HIGHLIGHT 2: NOISE AT THE END (ĐOẠN CUỐI) ---
    # Mũi tên chỉ vào Step 100.000, Loss ~0.15
    # Nếu dữ liệu của bạn ngắn hơn 100k step, nó sẽ tự điều chỉnh
    target_step = max_step * 0.8  # Lấy vị trí 80% quãng đường

    plt.annotate('Active Exploration\n(Higher Noise)',
                 xy=(target_step, 0.2),  # Điểm chỉ vào (Loss thấp ~0.2)
                 xytext=(target_step - 20000, 0.5),  # Chữ đặt cao hơn và lùi về trái
                 arrowprops=dict(facecolor='black', arrowstyle="->", lw=2),
                 fontsize=12, fontweight='bold', color='black',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig("plot_training_loss.png", dpi=300)
    print("✅ Đã tạo lại ảnh có mũi tên: plot_training_loss_highlight_fixed.png")


if __name__ == "__main__": plot_figure_2()