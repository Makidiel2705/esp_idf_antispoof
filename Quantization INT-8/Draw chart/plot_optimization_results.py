import matplotlib.pyplot as plt
import os

output_dir = r"d:\Project_TotNghiep\esp_idf_antispoof"

# Data for Model Size
model_versions = ['Float32 (Original)', 'Full INT8 (Quantized)']
sizes_mb = [1.64, 0.59]  # 1.64 MB vs 607 KB (~0.59 MB)

# Data for Processing Time
versions = ['Float32 (Gốc)', 'INT8 (Chưa tăng tốc)', 'INT8 + ESP-NN (Tối ưu)']
times_ms = [13200.0, 2840.0, 1106.0]  # 13.2s, 2.84s, 1.106s

# Color palette
colors_size = ['#34495E', '#2ECC71']
colors_time = ['#E74C3C', '#F39C12', '#2ECC71']

# Chart 1: Model Size
plt.figure(figsize=(6, 5), dpi=300)
bars = plt.bar(model_versions, sizes_mb, color=colors_size, width=0.45)
plt.title('So sanh dung luong tep tin mo hinh\n(Model Size Comparison)', fontsize=11, fontweight='bold', pad=15)
plt.ylabel('Dung luong (MB)', fontsize=10, labelpad=10)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f} MB', ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 2.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
size_chart_path = os.path.join(output_dir, 'model_size_comparison.png')
plt.savefig(size_chart_path, dpi=300)
plt.close()
print(f"Saved size chart to: {size_chart_path}")

# Chart 2: Processing Time
plt.figure(figsize=(7, 5), dpi=300)
bars = plt.bar(versions, [t/1000.0 for t in times_ms], color=colors_time, width=0.5)
plt.title('So sanh thoi gian xu ly suy luan\n(Inference Latency Comparison)', fontsize=11, fontweight='bold', pad=15)
plt.ylabel('Thoi gian (Giay)', fontsize=10, labelpad=10)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f} s', ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 15.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
time_chart_path = os.path.join(output_dir, 'processing_time_comparison.png')
plt.savefig(time_chart_path, dpi=300)
plt.close()
print(f"Saved time chart to: {time_chart_path}")

print("All charts generated successfully!")
