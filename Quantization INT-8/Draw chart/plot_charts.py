import os
import csv
import matplotlib.pyplot as plt

# Đường dẫn tệp tin
csv_path = r"d:\Project_TotNghiep\esp_idf_antispoof\antispoof_report.csv"
output_dir = r"d:\Project_TotNghiep\esp_idf_antispoof"

# Đọc dữ liệu từ CSV bằng thư viện chuẩn csv (tránh phụ thuộc vào pandas)
images = []
pytorch_vals = []
int8_vals = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    # Lấy chỉ số các cột
    py_idx = header.index("pytorch_real_percent")
    i8_idx = header.index("tflite_int8_real_percent")
    
    for row in reader:
        if not row:
            continue
        images.append(row[0])
        pytorch_vals.append(float(row[py_idx]))
        int8_vals.append(float(row[i8_idx]))

total_samples = len(pytorch_vals)

# Cấu hình font và style chung để biểu đồ trông hiện đại, chuyên nghiệp
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linewidth'] = 0.5

# ----------------------------------------------------
# BIỂU ĐỒ 1: Biểu đồ đường so sánh giá trị dự đoán (Line Chart)
# ----------------------------------------------------
plt.figure(figsize=(12, 5), dpi=300)
plt.plot(pytorch_vals, label="PyTorch (Float32)", color="#1F77B4", alpha=0.8, linewidth=1.5)
plt.plot(int8_vals, label="TFLite INT8 (Quantized)", color="#FF7F0E", alpha=0.8, linewidth=1.2, linestyle="--")
plt.title("So sanh xac suat du doan thuc the song (Liveness Probability)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Chi so mau anh (Sample Index)", labelpad=10)
plt.ylabel("Xac suat Real Face (%)", labelpad=10)
plt.xlim(0, total_samples)
plt.ylim(-5, 105)
plt.grid(True, linestyle=":")
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()
line_chart_path = os.path.join(output_dir, "liveness_comparison_line.png")
plt.savefig(line_chart_path, dpi=300)
plt.close()
print(f"Saved line chart to: {line_chart_path}")

# ----------------------------------------------------
# BIỂU ĐỒ 2: Biểu đồ phân tán tương quan (Scatter Plot)
# ----------------------------------------------------
plt.figure(figsize=(6.5, 6), dpi=300)
plt.scatter(pytorch_vals, int8_vals, color="#2CA02C", alpha=0.6, edgecolors='none', s=25, label="Mau thuc nghiem")
# Vẽ đường chéo y = x chuẩn
plt.plot([0, 100], [0, 100], color="#D62728", linestyle="--", linewidth=1.5, label="y = x (Ly tuong)")
plt.title("Do thi tuong quan xac suat giua Float32 va INT8", fontsize=11, fontweight='bold', pad=15)
plt.xlabel("Xac suat cua PyTorch Float32 (%)", labelpad=10)
plt.ylabel("Xac suat cua TFLite INT8 (%)", labelpad=10)
plt.xlim(-5, 105)
plt.ylim(-5, 105)
plt.grid(True, linestyle=":")
plt.legend(loc="upper left", frameon=True, facecolor="white")
plt.tight_layout()
scatter_plot_path = os.path.join(output_dir, "liveness_correlation_scatter.png")
plt.savefig(scatter_plot_path, dpi=300)
plt.close()
print(f"Saved scatter plot to: {scatter_plot_path}")

# ----------------------------------------------------
# BIỂU ĐỒ 3: Biểu đồ cột chồng phân loại nhãn và mẫu đảo ngược (Grouped-Stacked Bar Chart)
# ----------------------------------------------------
# Tính toán các mẫu đồng nhất và các mẫu bị đảo ngược nhãn (ngưỡng 50%)
py_real_consistent = 0
py_real_flipped = 0  # Real ở PyTorch nhưng Spoof ở INT8
py_spoof_consistent = 0
py_spoof_flipped = 0  # Spoof ở PyTorch nhưng Real ở INT8

i8_real_consistent = 0
i8_real_flipped = 0  # Real ở INT8 nhưng gốc là Spoof ở PyTorch
i8_spoof_consistent = 0
i8_spoof_flipped = 0  # Spoof ở INT8 nhưng gốc là Real ở PyTorch

for py, i8 in zip(pytorch_vals, int8_vals):
    py_is_real = py >= 50.0
    i8_is_real = i8 >= 50.0
    
    if py_is_real:
        if i8_is_real:
            py_real_consistent += 1
            i8_real_consistent += 1
        else:
            py_real_flipped += 1
            i8_spoof_flipped += 1
    else:
        if not i8_is_real:
            py_spoof_consistent += 1
            i8_spoof_consistent += 1
        else:
            py_spoof_flipped += 1
            i8_real_flipped += 1

categories = ['Real (Khuon mat that)', 'Spoof (Anh gia mao)']
x = [0, 1]
width = 0.35

fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

# Vẽ các cột cho PyTorch (Float32)
rects1_base = ax.bar([pos - width/2 for pos in x], [py_real_consistent, py_spoof_consistent], width, 
                      label='PyTorch - Dong nhat voi TFLite', color='#1F77B4')
rects1_flip = ax.bar([pos - width/2 for pos in x], [py_real_flipped, py_spoof_flipped], width, 
                      bottom=[py_real_consistent, py_spoof_consistent],
                      label='PyTorch - Khac biet so voi TFLite', color='#D62728', alpha=0.85)

# Vẽ các cột cho TFLite INT8 (Tham chiếu chuẩn - Flat)
i8_real_total = i8_real_consistent + i8_real_flipped
i8_spoof_total = i8_spoof_consistent + i8_spoof_flipped
rects2 = ax.bar([pos + width/2 for pos in x], [i8_real_total, i8_spoof_total], width, 
                 label='TFLite INT8 (Tham chieu chuan)', color='#FF7F0E')

ax.set_ylabel('So luong mau anh (Tam)')
ax.set_title('So sanh ket qua phan lop va ti le lech nhan (Nguong 50%)', fontsize=11, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(frameon=True, facecolor="white", loc="upper left")
ax.grid(True, axis='y', linestyle=":")

# Thêm nhãn cho cột chồng PyTorch
def autolabel_stacked(rects_base, rects_flip):
    for r_base, r_flip in zip(rects_base, rects_flip):
        h_base = r_base.get_height()
        h_flip = r_flip.get_height()
        total = h_base + h_flip
        
        # Nhãn tổng số lượng ở trên cùng
        ax.annotate(f'{total}',
                    xy=(r_flip.get_x() + r_flip.get_width() / 2, total),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Nhãn số lượng bị khác biệt ở bên trong cột đỏ
        if h_flip > 0:
            ax.annotate(f'{h_flip}',
                        xy=(r_flip.get_x() + r_flip.get_width() / 2, h_base + h_flip / 2),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Thêm nhãn cho cột phẳng TFLite
def autolabel_flat(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel_stacked(rects1_base, rects1_flip)
autolabel_flat(rects2)

plt.ylim(0, max(py_spoof_consistent + py_spoof_flipped, i8_spoof_consistent + i8_spoof_flipped) * 1.15)
plt.tight_layout()
bar_chart_path = os.path.join(output_dir, "liveness_classification_bar.png")
plt.savefig(bar_chart_path, dpi=300)
plt.close()
print(f"Saved stacked bar chart to: {bar_chart_path}")

# ----------------------------------------------------
# BIỂU ĐỒ 4: Biểu đồ tần suất phân phối sai số (Error Histogram)
# ----------------------------------------------------
# Tính sai số tuyệt đối
errors = [abs(py - i8) for py, i8 in zip(pytorch_vals, int8_vals)]

# Phân nhóm sai số
bins = [0, 2, 5, 10, 20, 50, 75, 100]
labels = ['0-2%', '2-5%', '5-10%', '10-20%', '20-50%', '50-75%', '>75%']
counts = [0] * len(labels)

for e in errors:
    for idx, limit in enumerate(bins[1:]):
        if e <= limit:
            counts[idx] += 1
            break

plt.figure(figsize=(7.5, 5.5), dpi=300)
bars = plt.bar(labels, counts, color='#9467BD', alpha=0.85, edgecolor='none', width=0.6)
plt.title("Phan phoi sai so tuyet doi do luong tu hoa INT8", fontsize=11, fontweight='bold', pad=15)
plt.xlabel("Khoang sai lech xac suat (Error Range)", labelpad=10)
plt.ylabel("So luong mau anh (Tam)", labelpad=10)
plt.grid(True, axis='y', linestyle=":")

# Thêm nhãn số lượng và phần trăm trên đầu cột
for bar in bars:
    height = bar.get_height()
    percentage = (height / total_samples) * 100
    plt.annotate(f'{height}\n({percentage:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.ylim(0, max(counts) * 1.15)
plt.tight_layout()
error_hist_path = os.path.join(output_dir, "liveness_error_histogram.png")
plt.savefig(error_hist_path, dpi=300)
plt.close()
print(f"Saved error histogram to: {error_hist_path}")

print("\n=== FINISHED ===")
print("All charts have been generated successfully.")
