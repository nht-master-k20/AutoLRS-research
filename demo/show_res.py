import pandas as pd
import numpy as np


def analyze_results():
    files = {
        "AutoLRS": "reproduce_vgg_log.csv",
        "Baseline": "baseline_vgg_log.csv",
        "Cosine": "cosine_vgg_log.csv"
    }

    print(f"{'ALGORITHM':<15} | {'BEST ACC (%)':<15} | {'BEST EPOCH':<12} | {'FINAL ACC (%)':<15}")
    print("-" * 65)

    for name, filepath in files.items():
        try:
            df = pd.read_csv(filepath)

            # Lọc dữ liệu validation (chỉ lấy dòng có Val_Acc > 0)
            val_df = df[pd.to_numeric(df['Val_Acc'], errors='coerce') > 0]

            if val_df.empty:
                print(f"{name:<15} | {'No Data':<15} | {'-':<12} | {'-':<15}")
                continue

            # Tìm Best Accuracy
            best_acc = val_df['Val_Acc'].max()
            best_epoch = val_df.loc[val_df['Val_Acc'].idxmax(), 'Epoch']

            # Tìm Final Accuracy (Acc tại epoch cuối cùng)
            final_acc = val_df.iloc[-1]['Val_Acc']

            print(f"{name:<15} | {best_acc:<15.2f} | {best_epoch:<12} | {final_acc:<15.2f}")

        except Exception as e:
            print(f"Lỗi đọc file {name}: {e}")


if __name__ == "__main__":
    analyze_results()