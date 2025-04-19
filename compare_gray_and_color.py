import numpy as np
from scipy.ndimage import median_filter
import cv2
import time
import matplotlib.pyplot as plt # Thư viện để vẽ đồ thị

def atpmf_grayscale(X, W1_size=3, W2_size=3, a=1.0, b=1.0):
    if X.ndim != 2: raise ValueError("Ảnh đầu vào phải là ảnh thang độ xám.")
    if W1_size % 2 == 0 or W2_size % 2 == 0: raise ValueError("Kích thước cửa sổ lọc phải là số lẻ.")
    X_float = X.astype(np.float64)
    M, N = X_float.shape
    Y = median_filter(X_float, size=W1_size, mode='reflect')
    E1 = (np.abs(X_float - Y) > 1e-6).astype(int)
    Y_tilde = Y.copy()
    E2 = np.zeros_like(X_float, dtype=int)
    if M * N > 0:
        lambda_n = np.sum(E1, axis=0) / M if M > 0 else np.zeros(N)
        Lambda = np.mean(lambda_n)
    else: Lambda = 0; lambda_n = np.zeros(N)
    sigma_lambda = np.std(lambda_n) if N > 1 else 0
    eta = a * sigma_lambda
    epsilon = 1e-9
    for n in range(N):
        if sigma_lambda > epsilon and (lambda_n[n] - Lambda) > eta:
            e_col = X_float[:, n] - Y[:, n]
            K = int(round(max(0.0, (lambda_n[n] - Lambda + b * sigma_lambda)) * M))
            if K > 0:
                changed_indices = np.where(E1[:, n] == 1)[0]
                if len(changed_indices) > 0:
                    abs_errors_at_changed = np.abs(e_col[changed_indices])
                    num_to_restore = min(K, len(changed_indices))
                    if num_to_restore > 0:
                        sorted_local_indices = np.argsort(abs_errors_at_changed)
                        indices_to_restore_local = sorted_local_indices[:num_to_restore]
                        indices_to_restore_global = changed_indices[indices_to_restore_local]
                        Y_tilde[indices_to_restore_global, n] = X_float[indices_to_restore_global, n]
                        E2[indices_to_restore_global, n] = 1
    Z_pass2 = median_filter(Y_tilde, size=W2_size, mode='reflect')
    Z = np.where(E2 == 1, Y_tilde, Z_pass2)
    if X.dtype == np.uint8: Z = np.clip(Z, 0, 255).astype(np.uint8)
    return Z

def atpmf_color_per_channel(X_color, W1_size=3, W2_size=3, a=1.0, b=1.0):
    if X_color.ndim != 3 or X_color.shape[2] != 3: raise ValueError("Ảnh đầu vào phải là ảnh màu 3 kênh.")
    original_dtype = X_color.dtype
    X_color_float = X_color.astype(np.float64)
    channels = cv2.split(X_color_float)
    processed_channels = []
    for channel in channels:
        processed_channel = atpmf_grayscale(channel, W1_size, W2_size, a, b)
        processed_channels.append(processed_channel)
    Z_color = cv2.merge(processed_channels)
    if original_dtype == np.uint8: Z_color = np.clip(Z_color, 0, 255).astype(np.uint8)
    return Z_color

def add_salt_and_pepper_noise(image, noise_ratio=0.1):
    noisy_image = np.copy(image)
    if image.ndim == 2: # Grayscale
        H, W = image.shape; num_pixels = H * W
        num_noise = int(noise_ratio * num_pixels); num_salt = num_noise // 2
        num_pepper = num_noise - num_salt
        salt_coords_h = np.random.randint(0, H, num_salt)
        salt_coords_w = np.random.randint(0, W, num_salt)
        noisy_image[salt_coords_h, salt_coords_w] = 255
        pepper_coords_h = np.random.randint(0, H, num_pepper)
        pepper_coords_w = np.random.randint(0, W, num_pepper)
        noisy_image[pepper_coords_h, pepper_coords_w] = 0
    elif image.ndim == 3 and image.shape[2] == 3: # Color
        H, W, C = image.shape; num_pixels = H * W
        num_noise = int(noise_ratio * num_pixels); num_salt = num_noise // 2
        num_pepper = num_noise - num_salt
        salt_coords_h = np.random.randint(0, H, num_salt)
        salt_coords_w = np.random.randint(0, W, num_salt)
        noisy_image[salt_coords_h, salt_coords_w, :] = 255
        pepper_coords_h = np.random.randint(0, H, num_pepper)
        pepper_coords_w = np.random.randint(0, W, num_pepper)
        noisy_image[pepper_coords_h, pepper_coords_w, :] = 0
    else: raise ValueError("Định dạng ảnh không hỗ trợ.")
    return noisy_image

def calculate_mae(image1, image2):
    if image1.shape != image2.shape: raise ValueError("Kích thước ảnh phải giống nhau.")
    img1_float = image1.astype(np.float64)
    img2_float = image2.astype(np.float64)
    return np.mean(np.abs(img1_float - img2_float))

def calculate_mse(image1, image2):
    if image1.shape != image2.shape: raise ValueError("Kích thước ảnh phải giống nhau.")
    img1_float = image1.astype(np.float64)
    img2_float = image2.astype(np.float64)
    return np.mean((img1_float - img2_float) ** 2)

if __name__ == "__main__":
    # --- Tham số ---
    grayscale_image_path = 'lena.jpg'
    color_image_path = 'lena_color.png'
    w1 = 3
    w2 = 3
    param_a = 1.0
    param_b = 1.0

    # --- Dải mức nhiễu cần kiểm tra (% -> ratio) ---
    noise_percentages = np.arange(5, 101, 5) # Từ 5% đến 100%, bước nhảy 5%
    noise_ratios = noise_percentages / 100.0

    # --- List để lưu kết quả ---
    mae_gray_results = []
    mse_gray_results = []
    mae_color_results = []
    mse_color_results = []

    print("Đang thực hiện tính toán lỗi MAE/MSE với các mức nhiễu khác nhau...")
    print(f"Ảnh thang xám: {grayscale_image_path}")
    print(f"Ảnh màu: {color_image_path}")
    print(f"Tham số ATPMF: W1={w1}, W2={w2}, a={param_a}, b={param_b}")
    print("-" * 50)

    # --- Tải ảnh gốc một lần ---
    try:
        original_gray = cv2.imread(grayscale_image_path, cv2.IMREAD_GRAYSCALE)
        original_color = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
        if original_gray is None: raise FileNotFoundError(f"Không tìm thấy hoặc không đọc được ảnh: {grayscale_image_path}")
        if original_color is None: raise FileNotFoundError(f"Không tìm thấy hoặc không đọc được ảnh: {color_image_path}")

        # --- Lặp qua từng mức nhiễu ---
        for i, noise_ratio in enumerate(noise_ratios):
            print(f"Đang xử lý mức nhiễu: {noise_percentages[i]}% (Ratio: {noise_ratio:.2f})")

            # 1. Xử lý ảnh thang xám
            noisy_gray = add_salt_and_pepper_noise(original_gray, noise_ratio=noise_ratio)
            filtered_gray = atpmf_grayscale(noisy_gray, W1_size=w1, W2_size=w2, a=param_a, b=param_b)
            mae_g = calculate_mae(original_gray, filtered_gray)
            mse_g = calculate_mse(original_gray, filtered_gray)
            mae_gray_results.append(mae_g)
            mse_gray_results.append(mse_g)
            print(f"  [Gray]  MAE: {mae_g:.4f}, MSE: {mse_g:.4f}")

            # 2. Xử lý ảnh màu
            noisy_color = add_salt_and_pepper_noise(original_color, noise_ratio=noise_ratio)
            filtered_color = atpmf_color_per_channel(noisy_color, W1_size=w1, W2_size=w2, a=param_a, b=param_b)
            mae_c = calculate_mae(original_color, filtered_color)
            mse_c = calculate_mse(original_color, filtered_color)
            mae_color_results.append(mae_c)
            mse_color_results.append(mse_c)
            print(f"  [Color] MAE: {mae_c:.4f}, MSE: {mse_c:.4f}")

        print("-" * 50)
        print("Hoàn thành tính toán. Đang vẽ biểu đồ...")

        # --- Vẽ biểu đồ ---
        plt.style.use('seaborn-v0_8-darkgrid') # Chọn style cho đẹp hơn (tùy chọn)

        # Biểu đồ MAE
        plt.figure(figsize=(10, 6))
        plt.plot(noise_percentages, mae_gray_results, marker='o', linestyle='-', label='Ảnh thang xám (Grayscale)')
        plt.plot(noise_percentages, mae_color_results, marker='s', linestyle='--', label='Ảnh màu (Color - Per Channel)')
        plt.title(f'So sánh MAE của ATPMF (W1={w1}, W2={w2}, a={param_a}, b={param_b})\nvới các mức nhiễu Salt & Pepper khác nhau')
        plt.xlabel('% Nhiễu Salt & Pepper')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xticks(noise_percentages) # Đảm bảo các tick đúng với % nhiễu
        plt.legend() # Hiển thị chú thích
        plt.grid(True) # Hiển thị lưới

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