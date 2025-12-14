import pandas as pd
import os


def analyze_results():
    files = {
        "AutoLRS": "reproduce_vgg_log.csv",
        "Baseline": "baseline_vgg_log.csv",
        "Cosine": "cosine_vgg_log.csv"
    }

    results = []

    for name, filepath in files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                val_df = df[pd.to_numeric(df['Val_Acc'], errors='coerce') > 0]

                if val_df.empty:
                    results.append((name, 0, 0, 0, 0))
                    continue

                best_acc = val_df['Val_Acc'].max()
                # Lấy epoch tương ứng với Best Acc
                best_epoch = val_df.loc[val_df['Val_Acc'].idxmax(), 'Epoch']
                final_acc = val_df.iloc[-1]['Val_Acc']
                total_epochs = val_df['Epoch'].max()

                results.append((name, best_acc, best_epoch, final_acc, total_epochs))
            except:
                pass

    # Sắp xếp theo Best Acc giảm dần
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 85)
    print(f"{'ALGORITHM':<15} | {'BEST ACC (%)':<15} | {'AT EPOCH':<12} | {'FINAL ACC (%)':<15} | {'TOTAL EPOCHS':<12}")
    print("-" * 85)

    for row in results:
        name, best, epoch, final, total = row
        print(f"{name:<15} | {best:<15.2f} | {int(epoch):<12} | {final:<15.2f} | {int(total):<12}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    analyze_results()