import pandas as pd
import numpy as np
import os


def analyze_full_metrics():
    files = {
        "AutoLRS": "reproduce_vgg_log.csv",
        "Baseline": "baseline_vgg_log.csv",
        "Cosine": "cosine_vgg_log.csv"
    }

    # Cấu hình ngưỡng hội tụ (ví dụ: mất bao lâu để đạt 90%)
    TARGET_ACC_THRESHOLD = 90.0

    stats = []

    print("\n" + "=" * 125)
    print(f"{'COMPREHENSIVE COMPARISON':^125}")
    print("=" * 125)

    # Header bảng
    header = f"{'ALGORITHM':<12} | {'BEST ACC':<10} | {'MIN VAL LOSS':<12} | {'EPOCH >90%':<10} | {'TOTAL TIME (h)':<14} | {'AVG TIME/EP (s)':<15} | {'TRAIN LOSS':<10}"
    print(header)
    print("-" * 125)

    for name, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"{name:<12} | File Not Found")
            continue

        try:
            # Đọc file, bỏ qua dòng lỗi
            df = pd.read_csv(filepath, on_bad_lines='skip')

            # --- 1. CLEAN DATA ---
            cols = ['Epoch', 'Time', 'Val_Acc', 'Val_Loss', 'Train_Loss']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            # Tách dữ liệu Validation (cuối epoch) và Training (trong epoch)
            val_df = df[df['Val_Acc'] > 0].copy()
            train_df = df[df['Train_Loss'].notnull()].copy()

            if val_df.empty:
                print(f"{name:<12} | No Validation Data")
                continue

            # --- 2. CALCULATE METRICS ---

            # a. Performance
            best_acc = val_df['Val_Acc'].max()
            min_val_loss = val_df['Val_Loss'].min()
            final_train_loss = train_df['Train_Loss'].iloc[-1] if not train_df.empty else 0

            # b. Convergence Speed (Tìm epoch đầu tiên vượt ngưỡng 90%)
            converge_df = val_df[val_df['Val_Acc'] >= TARGET_ACC_THRESHOLD]
            if not converge_df.empty:
                epoch_90 = int(converge_df['Epoch'].min())
                speed_str = f"{epoch_90}"
            else:
                speed_str = ">Max"

            # c. Time Efficiency
            # Lấy max time (giây) chia 3600 ra giờ
            total_time_sec = df['Time'].max()
            total_time_hours = total_time_sec / 3600

            total_epochs = val_df['Epoch'].max()
            avg_time_per_epoch = total_time_sec / total_epochs if total_epochs > 0 else 0

            stats.append({
                "name": name,
                "best_acc": best_acc,
                "min_val_loss": min_val_loss,
                "speed": speed_str,
                "time_h": total_time_hours,
                "time_ep": avg_time_per_epoch,
                "train_loss": final_train_loss
            })

        except Exception as e:
            print(f"{name:<12} | Error: {e}")

    # Sắp xếp theo Best Accuracy giảm dần
    stats.sort(key=lambda x: x['best_acc'], reverse=True)

    # In ra bảng
    for s in stats:
        print(
            f"{s['name']:<12} | {s['best_acc']:<10.2f} | {s['min_val_loss']:<12.4f} | {s['speed']:<10} | {s['time_h']:<14.2f} | {s['time_ep']:<15.2f} | {s['train_loss']:<10.4f}")

    print("-" * 125)
    print("(*) EPOCH >90%: Số epoch cần để đạt độ chính xác 90%. Càng nhỏ = Hội tụ càng nhanh.")
    print("(*) AVG TIME/EP: Thời gian trung bình 1 Epoch. Dùng để đo chi phí tính toán của thuật toán.")
    print("(*) MIN VAL LOSS: Giá trị Loss thấp nhất trên tập kiểm thử (Càng thấp càng tốt).")
    print("=" * 125 + "\n")


if __name__ == "__main__":
    analyze_full_metrics()