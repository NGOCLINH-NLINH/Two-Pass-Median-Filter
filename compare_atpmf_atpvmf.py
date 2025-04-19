# compare_filters.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm # Để theo dõi tiến trình

# Import các hàm từ các file của bạn
# Đảm bảo atpmf.py và color_filter.py nằm cùng thư mục
try:
    from atpmf import atpmf_color_per_channel
    print("Đã import thành công atpmf_color_per_channel từ atpmf.py")
except ImportError:
    print("Lỗi: Không thể import atpmf_color_per_channel. Đảm bảo file atpmf.py tồn tại và không có lỗi.")
    exit()
except Exception as e:
    print(f"Lỗi khi import từ atpmf.py: {e}")
    exit()

try:
    # Đảm bảo import đúng hàm đã tối ưu
    from color_filter import adaptive_two_pass_vmf_optimized
    print("Đã import thành công adaptive_two_pass_vmf_optimized từ color_filter.py")
except ImportError:
    print("Lỗi: Không thể import adaptive_two_pass_vmf_optimized. Đảm bảo file color_filter.py tồn tại và không có lỗi.")
    exit()
except Exception as e:
    print(f"Lỗi khi import từ color_filter.py: {e}")
    exit()

# --- Hàm Helper ---

def add_salt_pepper_color(image, noise_percentage):
    """
    Thêm nhiễu Salt & Pepper vào ảnh màu.

    Tham số:
        image (numpy.ndarray): Ảnh màu đầu vào (BGR, uint8).
        noise_percentage (float): Tỷ lệ phần trăm tổng số pixel bị nhiễu (0.0 đến 1.0).

    Trả về:
        numpy.ndarray: Ảnh màu bị nhiễu (uint8).
    """
    noisy_image = np.copy(image)
    height, width, _ = image.shape
    num_pixels = height * width
    num_noisy_pixels = int(noise_percentage * num_pixels)

    # Chia đều cho salt và pepper
    num_salt = num_noisy_pixels // 2
    num_pepper = num_noisy_pixels - num_salt

    # Thêm nhiễu Salt (trắng)
    # Chọn ngẫu nhiên các tọa độ pixel (không lặp lại)
    salt_coords_flat = np.random.choice(num_pixels, num_salt, replace=False)
    salt_rows = salt_coords_flat // width
    salt_cols = salt_coords_flat % width
    noisy_image[salt_rows, salt_cols, :] = 255 # Đặt cả 3 kênh thành trắng

    # Thêm nhiễu Pepper (đen)
    # Chọn ngẫu nhiên các tọa độ pixel khác (không trùng với salt)
    # Tạo tập hợp tất cả các chỉ số pixel
    all_coords_flat = np.arange(num_pixels)
    # Loại bỏ các chỉ số đã dùng cho salt
    available_for_pepper = np.setdiff1d(all_coords_flat, salt_coords_flat, assume_unique=True)
    # Chọn từ các chỉ số còn lại
    pepper_coords_flat = np.random.choice(available_for_pepper, num_pepper, replace=False)
    pepper_rows = pepper_coords_flat // width
    pepper_cols = pepper_coords_flat % width
    noisy_image[pepper_rows, pepper_cols, :] = 0 # Đặt cả 3 kênh thành đen

    return noisy_image.astype(image.dtype)

def calculate_mae(image1, image2):
    """
    Tính toán Mean Absolute Error (MAE) giữa hai ảnh.
    """
    if image1.shape != image2.shape:
        raise ValueError("Kích thước hai ảnh phải giống nhau để tính MAE.")
    # Chuyển sang float để tránh lỗi tràn số khi trừ
    img1_f = image1.astype(np.float64)
    img2_f = image2.astype(np.float64)
    mae = np.mean(np.abs(img1_f - img2_f))
    return mae

# --- Cấu hình ---
ORIGINAL_IMAGE_PATH = 'anhdomixi.png' # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY
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
    noisy_image = add_salt_pepper_color(original_image, noise_perc)
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
plt.xticks(NOISE_PERCENTAGES * 100) # Đảm bảo hiển thị các mốc %
plt.grid(True)
plt.legend()
plt.show()

# # --- Vẽ biểu đồ so sánh thời gian xử lý ---
# plt.figure(figsize=(10, 6))
# plt.plot(NOISE_PERCENTAGES * 100, processing_times_atpmf, marker='o', linestyle='-', label='ATPMF (Per Channel)')
# plt.plot(NOISE_PERCENTAGES * 100, processing_times_atvmf, marker='s', linestyle='--', label='Adaptive VMF (Optimized)')
#
# plt.title('So sánh Thời gian xử lý của ATPMF và Adaptive VMF')
# plt.xlabel('Tỷ lệ nhiễu Salt & Pepper (%)')
# plt.ylabel('Thời gian xử lý (giây)')
# plt.xticks(NOISE_PERCENTAGES * 100)
# plt.grid(True)
# plt.legend()
# plt.show()

print("\nHoàn thành so sánh.")