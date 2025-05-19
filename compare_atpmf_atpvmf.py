# compare_filters.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm # Để theo dõi tiến trình
from atpmf import atpmf_color_per_channel
from color_filter import adaptive_two_pass_vmf_optimized
from utils import add_salt_and_pepper_noise
from utils import calculate_mse
from utils import calculate_mae

# --- Cấu hình ---
ORIGINAL_IMAGE_PATH = 'images/lena_color.png'  # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY
NOISE_PERCENTAGES = np.arange(0.05, 1.01, 0.05) # 5%, 10%, ..., 100%

# Tham số cho bộ lọc (có thể điều chỉnh nếu cần)
W1_SIZE = 3
W2_SIZE = 3
PARAM_A = 1.0
PARAM_B = 1.0

# --- Thực hiện so sánh ---
print(f"Đang nạp ảnh gốc: {ORIGINAL_IMAGE_PATH}")
original_image = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_COLOR)

if original_image is None:
    print(f"Lỗi: Không thể đọc ảnh từ '{ORIGINAL_IMAGE_PATH}'. Vui lòng kiểm tra đường dẫn.")
    exit()

print(f"Kích thước ảnh gốc: {original_image.shape}")

mae_results_atpmf = []
mae_results_atvmf = []
processing_times_atpmf = []
processing_times_atvmf = []

print("\nBắt đầu quá trình thêm nhiễu và lọc ảnh...")

for noise_perc in tqdm(NOISE_PERCENTAGES, desc="Processing Noise Levels"):
    print(f"\n--- Đang xử lý mức nhiễu: {noise_perc*100:.0f}% ---")

    # 1. Thêm nhiễu Salt & Pepper
    noisy_image = add_salt_and_pepper_noise(original_image, noise_perc)
    # (Tùy chọn) Lưu ảnh nhiễu để kiểm tra
    # cv2.imwrite(f"temp_noisy_{int(noise_perc*100)}.png", noisy_image)

    # --- 2. Áp dụng ATPMF (per channel) ---
    print("Đang áp dụng ATPMF (per channel)...")
    start_time = time.time()
    # Hàm atpmf_color_per_channel tự xử lý kiểu dữ liệu bên trong
    filtered_atpmf = atpmf_color_per_channel(noisy_image,
                                             W1_size=W1_SIZE,
                                             W2_size=W2_SIZE,
                                             a=PARAM_A,
                                             b=PARAM_B)
    end_time = time.time()
    time_atpmf = end_time - start_time
    processing_times_atpmf.append(time_atpmf)
    print(f"ATPMF hoàn thành trong {time_atpmf:.4f} giây.")

    # --- 3. Áp dụng Adaptive Two Pass VMF (Optimized) ---
    print("Đang áp dụng Adaptive Two Pass VMF (Optimized)...")
    start_time = time.time()
    # Hàm ATVMF tối ưu yêu cầu đầu vào là float64
    filtered_atvmf = adaptive_two_pass_vmf_optimized(noisy_image.astype(np.float64),
                                                    W1_size=W1_SIZE,
                                                    W2_size=W2_SIZE,
                                                    a=PARAM_A,
                                                    b=PARAM_B)
    end_time = time.time()
    time_atvmf = end_time - start_time
    processing_times_atvmf.append(time_atvmf)
    print(f"ATVMF hoàn thành trong {time_atvmf:.4f} giây.")

    # --- 4. Tính toán MAE ---
    mae_atpmf = calculate_mae(original_image, filtered_atpmf)
    mae_atvmf = calculate_mae(original_image, filtered_atvmf)

    mae_results_atpmf.append(mae_atpmf)
    mae_results_atvmf.append(mae_atvmf)

    print(f"MAE (ATPMF): {mae_atpmf:.4f}")
    print(f"MAE (ATVMF): {mae_atvmf:.4f}")

# --- Hiển thị kết quả ---
print("\n--- Kết quả MAE tổng hợp ---")
print("-------------------------------------------")
print("| Mức nhiễu (%) | MAE (ATPMF) | MAE (ATVMF) |")
print("-------------------------------------------")
for i, noise_perc in enumerate(NOISE_PERCENTAGES):
    print(f"| {noise_perc*100:13.0f} | {mae_results_atpmf[i]:11.4f} | {mae_results_atvmf[i]:11.4f} |")
print("-------------------------------------------")

print("\n--- Thời gian xử lý trung bình ---")
print(f"ATPMF (per channel): {np.mean(processing_times_atpmf):.4f} giây")
print(f"ATVMF (optimized)  : {np.mean(processing_times_atvmf):.4f} giây")


# --- Vẽ biểu đồ so sánh MAE ---
plt.figure(figsize=(10, 6))
plt.plot(NOISE_PERCENTAGES * 100, mae_results_atpmf, marker='o', linestyle='-', label='ATPMF (Per Channel)')
plt.plot(NOISE_PERCENTAGES * 100, mae_results_atvmf, marker='s', linestyle='--', label='Adaptive VMF (Optimized)')

plt.title('So sánh MAE của ATPMF và Adaptive VMF với nhiễu Salt & Pepper')
plt.xlabel('Tỷ lệ nhiễu Salt & Pepper (%)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xticks(NOISE_PERCENTAGES * 100)
plt.grid(True)
plt.legend()
plt.show()

print("\nHoàn thành so sánh.")