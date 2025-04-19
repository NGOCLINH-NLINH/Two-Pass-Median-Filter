import numpy as np
# from scipy.ndimage import generic_filter # Không dùng nữa
from tqdm import tqdm
import cv2
import numba # Thêm thư viện Numba
import time # Để đo thời gian

@numba.njit(parallel=True, fastmath=True) # Bật song song và tối ưu toán học
def apply_vmf_numba_core(padded_image, output_image, window_size, height, width, channels):
    """
    Hàm Numba lõi để thực hiện VMF, tối ưu hóa và song song hóa.
    Lưu ý: Không dùng tqdm bên trong Numba.
    """
    pad_size = (window_size - 1) // 2
    window_num_pixels = window_size * window_size

    # Song song hóa vòng lặp ngoài cùng (r)
    for r in numba.prange(height):
        for c in range(width):
            # Trích xuất cửa sổ (dạng 3D: win_h, win_w, channels)
            # Không tạo biến window riêng để tránh cấp phát bộ nhớ thừa trong Numba
            # window = padded_image[r: r + window_size, c: c + window_size, :]

            # Tính toán trực tiếp trên cửa sổ con (view) của padded_image
            min_sum_sq_dist = np.inf
            median_vector = np.zeros(channels, dtype=np.float64) # Khởi tạo

            # Lấy pixel đầu tiên làm median ban đầu (tạm thời)
            # Phải truy cập trực tiếp padded_image thay vì window
            for ch in range(channels):
                 median_vector[ch] = padded_image[r + pad_size, c + pad_size, ch]


            # Duyệt qua các pixel 'candidate' trong cửa sổ
            for i_row in range(window_size):
                for i_col in range(window_size):
                    # candidate = window[i_row, i_col, :]
                    candidate = padded_image[r + i_row, c + i_col, :] # Lấy candidate trực tiếp

                    current_sum_sq_dist = 0.0
                    # Tính tổng khoảng cách từ candidate đến tất cả các pixel khác 'p' trong cửa sổ
                    for j_row in range(window_size):
                        for j_col in range(window_size):
                            # p = window[j_row, j_col, :]
                            p = padded_image[r + j_row, c + j_col, :] # Lấy p trực tiếp

                            # Tính bình phương khoảng cách L2 giữa candidate và p
                            sq_dist = 0.0
                            for ch in range(channels):
                                diff = candidate[ch] - p[ch]
                                sq_dist += diff * diff
                            current_sum_sq_dist += sq_dist

                    # Cập nhật median nếu tìm thấy tổng khoảng cách nhỏ hơn
                    if current_sum_sq_dist < min_sum_sq_dist:
                        min_sum_sq_dist = current_sum_sq_dist
                        # median_vector = candidate # Gán trực tiếp không an toàn trong Numba? Copy giá trị.
                        for ch in range(channels):
                            median_vector[ch] = candidate[ch]

            # Gán vector trung vị tìm được vào ảnh output
            output_image[r, c, :] = median_vector
    # Numba function không cần return vì nó sửa đổi output_image trực tiếp

def apply_vmf_optimized(image, window_size):
    """
    Hàm bao bọc để gọi Numba core cho VMF.
    """
    height, width, channels = image.shape
    pad_size = (window_size - 1) // 2
    # Quan trọng: Khởi tạo output_image với kiểu float64 ngay từ đầu
    output_image = np.zeros_like(image, dtype=np.float64)

    # Thêm padding - Nên làm với float64 để nhất quán
    padded_image = np.pad(image.astype(np.float64),
                          ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                          mode='edge')

    print(f"Applying Optimized VMF (Numba) with window {window_size}x{window_size}...")
    start_time = time.time()
    # Gọi hàm Numba đã biên dịch
    apply_vmf_numba_core(padded_image, output_image, window_size, height, width, channels)
    end_time = time.time()
    print(f"Optimized VMF Pass completed in {end_time - start_time:.4f} seconds.")

    # Không cần chuyển kiểu ở đây, giữ float64 cho các bước tiếp theo
    return output_image


# ==========================================================
#   ADAPTIVE TWO PASS VMF VỚI CORE VMF ĐÃ TỐI ƯU
# ==========================================================
def adaptive_two_pass_vmf_optimized(noisy_image, W1_size=3, W2_size=3, a=1.0, b=1.0):
    """
    Thực hiện thuật toán Adaptive Two-Pass Vector Median Filter TỐI ƯU HÓA.
    Sử dụng apply_vmf_optimized (Numba) và vector hóa Bước 2.
    """
    # Đảm bảo đầu vào là float64 ngay từ đầu
    X = noisy_image.astype(np.float64)
    M, N, C = X.shape

    # --- Step 1: Lọc VMF lần 1 (Sử dụng hàm tối ưu) ---
    print("Step 1: First Optimized VMF Pass")
    Y = apply_vmf_optimized(X, W1_size) # Gọi hàm đã tối ưu

    # Tính toán ma trận lỗi E1 (Giữ nguyên - đã khá nhanh)
    difference_norm = np.linalg.norm(X - Y, axis=2)
    E1 = (difference_norm > 1e-6).astype(int)

    # --- Step 2: Hiệu chỉnh thích ứng (Tối ưu vòng lặp khôi phục) ---
    print("Step 2: Adaptive Correction (Optimized Revert Loop)")
    # Tính toán thống kê (Giữ nguyên - đã khá nhanh)
    lambda_count_per_column = np.sum(E1, axis=0)
    Lambda_avg_count_per_column = np.sum(E1) / N if N > 0 else 0
    sigma_lambda_count = np.std(lambda_count_per_column) if N > 1 else 0
    eta_count = a * sigma_lambda_count

    Y_tilde = Y.copy()
    E2 = np.zeros_like(E1, dtype=int)

    print("Checking columns for over-correction (using counts)...")
    start_time_step2 = time.time()
    # Duyệt qua từng cột (vòng lặp này thường không quá lớn)
    for n in tqdm(range(N)): # Có thể giữ tqdm ở đây vì vòng lặp cột không quá nặng
        if sigma_lambda_count > 1e-9 and lambda_count_per_column[n] > Lambda_avg_count_per_column + eta_count:
            # Tính K
            excess_pixels = lambda_count_per_column[n] - Lambda_avg_count_per_column
            # Heuristic cho K (có thể cần tinh chỉnh)
            # K = round(excess_pixels + b * sigma_lambda_count * (lambda_count_per_column[n] / M if M > 0 else 0) )
            # Công thức đơn giản hơn từ paper gốc (cho tỷ lệ, điều chỉnh lại):
            # K = round(max(0.0, (lambda_count_per_column[n]/M - Lambda_avg_count_per_column/M + b * sigma_lambda_count/M)) * M)
            # Thử lại công thức ban đầu nhưng với counts:
            K = round(excess_pixels + b * sigma_lambda_count)
            K = max(0, min(K, int(lambda_count_per_column[n])))

            if K > 0:
                # Tính toán chênh lệch (nhanh)
                diff_vectors_col = X[:, n, :] - Y[:, n, :]
                diff_magnitudes_col = np.linalg.norm(diff_vectors_col, axis=1)
                changed_indices_in_col = np.where(E1[:, n] == 1)[0]

                if len(changed_indices_in_col) > 0:
                    magnitudes_of_changed = diff_magnitudes_col[changed_indices_in_col]
                    sorted_relative_indices = np.argsort(magnitudes_of_changed)
                    num_to_revert = min(K, len(sorted_relative_indices))
                    absolute_indices_to_revert = changed_indices_in_col[sorted_relative_indices[:num_to_revert]]

                    # --- TỐI ƯU HÓA: VECTOR HÓA PHẦN KHÔI PHỤC ---
                    # Thay vì dùng vòng lặp for m:
                    # for m in absolute_indices_to_revert:
                    #     Y_tilde[m, n, :] = X[m, n, :]
                    #     E2[m, n] = 1
                    # Dùng chỉ số mảng trực tiếp:
                    Y_tilde[absolute_indices_to_revert, n, :] = X[absolute_indices_to_revert, n, :]
                    E2[absolute_indices_to_revert, n] = 1
                    # -----------------------------------------------

    end_time_step2 = time.time()
    print(f"Adaptive Correction loop completed in {end_time_step2 - start_time_step2:.4f} seconds.")
    # --- Kết thúc Step 2 ---

    # --- Step 3: Lọc VMF lần 2 (Sử dụng hàm tối ưu) ---
    print("Step 3: Second Optimized VMF Pass with correction")
    Z_filtered = apply_vmf_optimized(Y_tilde, W2_size) # Gọi hàm đã tối ưu

    # Tạo ảnh kết quả cuối cùng Z (phần này giữ nguyên - đã nhanh)
    Z = Z_filtered # Khởi tạo từ kết quả lọc 2

    # Đảm bảo những pixel đã được khôi phục ở Bước 2 giữ nguyên giá trị gốc X
    revert_indices = np.where(E2 == 1)
    # Vector hóa việc copy giá trị cuối cùng
    Z[revert_indices] = X[revert_indices] # Copy trực tiếp tại các vị trí E2=1

    # Chuyển về kiểu dữ liệu gốc và clip giá trị
    # Quan trọng: Chuyển về uint8 chỉ ở bước cuối cùng
    Z = np.clip(Z, 0, 255).astype(np.uint8)

    return Z


# --- Ví dụ sử dụng (Giữ nguyên phần tạo nhiễu và gọi hàm) ---
if __name__ == "__main__":
    # Đọc ảnh màu
    input_image_path = 'images.jpg' # Thay bằng đường dẫn ảnh nhiễu

    # (Phần tạo nhiễu giữ nguyên nếu cần)
    # ... (code tạo noisy_image_sp) ...
    # cv2.imwrite(input_image_path, noisy_image_sp)

    noisy_image = cv2.imread(input_image_path)

    if noisy_image is None:
        print(f"Lỗi: Không thể đọc ảnh nhiễu tại '{input_image_path}'")
    else:
        print(f"Đã đọc ảnh nhiễu: {input_image_path}, kích thước: {noisy_image.shape}")

        print("\n--- Chạy thuật toán GỐC (để so sánh thời gian nếu cần) ---")
        start_original = time.time()
        # filtered_image_original = adaptive_two_pass_vmf(noisy_image.astype(np.float64), W1_size=3, W2_size=3, a=1.0, b=1.0) # Hàm gốc
        end_original = time.time()
        # print(f"Thời gian chạy thuật toán GỐC: {end_original - start_original:.4f} giây")


        print("\n--- Chạy thuật toán TỐI ƯU HÓA ---")
        start_optimized = time.time()
        # Chạy thuật toán lọc TỐI ƯU HÓA
        filtered_image_optimized = adaptive_two_pass_vmf_optimized(noisy_image.astype(np.float64),
                                                                  W1_size=3,
                                                                  W2_size=3,
                                                                  a=1.0,
                                                                  b=1.0)
        end_optimized = time.time()
        print(f"\nThời gian chạy thuật toán TỐI ƯU HÓA: {end_optimized - start_optimized:.4f} giây")

        # Lưu ảnh kết quả
        output_image_path_optimized = 'filtered_adaptive_vmf_optimized.png'
        cv2.imwrite(output_image_path_optimized, filtered_image_optimized)
        print(f"Đã lọc và lưu ảnh kết quả tối ưu: {output_image_path_optimized}")

        # (So sánh thời gian nếu chạy cả 2)
        # if 'filtered_image_original' in locals():
        #    speedup = (end_original - start_original) / (end_optimized - start_optimized)
        #    print(f"\nTốc độ tăng ~ {speedup:.2f} lần")

        # Hiển thị ảnh (tùy chọn)
        cv2.imshow('Noisy Image', noisy_image)
        cv2.imshow('Filtered Image (Optimized Adaptive VMF)', filtered_image_optimized)
        print("\nNhấn phím bất kỳ để đóng cửa sổ ảnh...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()