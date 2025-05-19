import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import add_salt_and_pepper_noise
from atpmf import atpmf_color_per_channel
from atpmf import atpmf_grayscale
from utils import calculate_mse
from utils import calculate_mae
from utils import convert_to_grayscale

if __name__ == "__main__":
    # --- Tham số ---
    original_image_path = 'images/lena_color.png'
    w1 = 3
    w2 = 3
    param_a = 1.0
    param_b = 1.0

    # --- Dải mức nhiễu cần kiểm tra (% -> ratio) ---
    noise_percentages = np.arange(0.05, 1.01, 0.05) # Từ 5% đến 100%, bước nhảy 5%

    # --- List để lưu kết quả ---
    mae_gray_results = []
    mse_gray_results = []
    mae_color_results = []
    mse_color_results = []

    print("Đang thực hiện tính toán lỗi MAE/MSE với các mức nhiễu khác nhau...")
    print(f"Ảnh gốc: {original_image_path}")
    print(f"Tham số ATPMF: W1={w1}, W2={w2}, a={param_a}, b={param_b}")
    print("-" * 50)

    # --- Tải ảnh gốc một lần ---
    try:
        original_color = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
        if original_color is None:
            raise FileNotFoundError(f"Không tìm thấy hoặc không đọc được ảnh: {original_image_path}")

        # Tạo ảnh grayscale từ ảnh màu
        original_gray = convert_to_grayscale(original_color)

        # --- Hiển thị ảnh trước và sau cho một mức nhiễu cụ thể ---
        # Chọn một mức nhiễu để hiển thị, ví dụ: 20%
        display_noise_percentage = 0.20
        # Tìm mức nhiễu gần nhất trong danh sách nếu 0.20 không có sẵn
        if display_noise_percentage not in noise_percentages:
            display_noise_idx = min(range(len(noise_percentages)), key=lambda i: abs(noise_percentages[i]-display_noise_percentage))
            display_noise_percentage = noise_percentages[display_noise_idx]

        print(f"\n--- Hiển thị ảnh cho mức nhiễu: {display_noise_percentage*100:.1f}% ---")

        # 1. Xử lý và hiển thị ảnh thang xám cho mức nhiễu đã chọn
        noisy_gray_display = add_salt_and_pepper_noise(original_gray, noise_percentage=display_noise_percentage)
        filtered_gray_display = atpmf_grayscale(noisy_gray_display, W1_size=w1, W2_size=w2, a=param_a, b=param_b)

        plt.figure(figsize=(15, 5))
        plt.suptitle(f'Ảnh Thang Xám (Grayscale) - Nhiễu: {display_noise_percentage*100:.1f}%', fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(original_gray, cmap='gray')
        plt.title('Ảnh Gốc (Grayscale)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(noisy_gray_display, cmap='gray')
        plt.title('Ảnh Nhiễu (Grayscale)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(filtered_gray_display, cmap='gray')
        plt.title('Ảnh Lọc (Grayscale ATPMF)')
        plt.axis('off')
        plt.show() # Hiển thị cửa sổ so sánh ảnh thang xám

        # 2. Xử lý và hiển thị ảnh màu cho mức nhiễu đã chọn
        # Thêm nhiễu vào bản sao của ảnh gốc màu để không ảnh hưởng tới các vòng lặp sau
        noisy_color_display = add_salt_and_pepper_noise(original_color.copy(), noise_percentage=display_noise_percentage)
        filtered_color_display = atpmf_color_per_channel(noisy_color_display, W1_size=w1, W2_size=w2, a=param_a, b=param_b)

        plt.figure(figsize=(18, 6))
        plt.suptitle(f'Ảnh Màu (Color) - Nhiễu: {display_noise_percentage*100:.1f}%', fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB))
        plt.title('Ảnh Gốc (Color)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(noisy_color_display, cv2.COLOR_BGR2RGB))
        plt.title('Ảnh Nhiễu (Color)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(filtered_color_display, cv2.COLOR_BGR2RGB))
        plt.title('Ảnh Lọc (Color ATPMF)')
        plt.axis('off')
        plt.show() # Hiển thị cửa sổ so sánh ảnh màu

        # --- Lặp qua từng mức nhiễu ---
        for i, noise_percentage in enumerate(noise_percentages):
            print(f"Đang xử lý mức nhiễu: {noise_percentages[i]*100}%")

            # 1. Xử lý ảnh thang xám
            noisy_gray = add_salt_and_pepper_noise(original_gray, noise_percentage=noise_percentage)
            filtered_gray = atpmf_grayscale(noisy_gray, W1_size=w1, W2_size=w2, a=param_a, b=param_b)
            mae_g = calculate_mae(original_gray, filtered_gray)
            mse_g = calculate_mse(original_gray, filtered_gray)
            mae_gray_results.append(mae_g)
            mse_gray_results.append(mse_g)
            print(f"  [Gray]  MAE: {mae_g:.4f}, MSE: {mse_g:.4f}")

            # 2. Xử lý ảnh màu
            noisy_color = add_salt_and_pepper_noise(original_color, noise_percentage=noise_percentage)
            filtered_color = atpmf_color_per_channel(noisy_color, W1_size=w1, W2_size=w2, a=param_a, b=param_b)
            mae_c = calculate_mae(original_color, filtered_color)
            mse_c = calculate_mse(original_color, filtered_color)
            mae_color_results.append(mae_c)
            mse_color_results.append(mse_c)
            print(f"  [Color] MAE: {mae_c:.4f}, MSE: {mse_c:.4f}")

        print("-" * 50)
        print("Hoàn thành tính toán. Đang vẽ biểu đồ...")

        # --- Vẽ biểu đồ ---
        plt.style.use('seaborn-v0_8-darkgrid')

        # Biểu đồ MAE
        plt.figure(figsize=(10, 6))
        plt.plot(noise_percentages, mae_gray_results, marker='o', linestyle='-', label='Ảnh thang xám (Grayscale)')
        plt.plot(noise_percentages, mae_color_results, marker='s', linestyle='--', label='Ảnh màu (Color - Per Channel)')
        plt.title(f'So sánh MAE của ATPMF (W1={w1}, W2={w2}, a={param_a}, b={param_b})\nvới các mức nhiễu Salt & Pepper khác nhau')
        plt.xlabel('% Nhiễu Salt & Pepper')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xticks(noise_percentages)
        plt.legend()
        plt.grid(True)

        # Biểu đồ MSE
        plt.figure(figsize=(10, 6))
        plt.plot(noise_percentages, mse_gray_results, marker='o', linestyle='-', label='Ảnh thang xám (Grayscale)')
        plt.plot(noise_percentages, mse_color_results, marker='s', linestyle='--', label='Ảnh màu (Color - Per Channel)')
        plt.title(f'So sánh MSE của ATPMF (W1={w1}, W2={w2}, a={param_a}, b={param_b})\nvới các mức nhiễu Salt & Pepper khác nhau')
        plt.xlabel('% Nhiễu Salt & Pepper')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.xticks(noise_percentages)
        plt.legend()
        plt.grid(True)

        plt.tight_layout() # Tự động điều chỉnh layout
        plt.show() # Hiển thị cả 2 biểu đồ

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        print("Vui lòng kiểm tra lại đường dẫn ảnh.")
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")