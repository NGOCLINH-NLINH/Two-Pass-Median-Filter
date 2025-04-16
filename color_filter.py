import numpy as np
from scipy.ndimage import generic_filter
from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình (tùy chọn)
import cv2  # Sử dụng OpenCV để đọc/ghi ảnh


def vector_median_filter_pixel(window_data):
    """
    Tính toán bộ lọc trung vị vector cho một cửa sổ (patch).
    window_data: Dữ liệu phẳng (flattened) của cửa sổ, shape (num_pixels * num_channels).
                 Ví dụ: cho cửa sổ 3x3 và ảnh RGB, shape là (9*3=27).
                 Cần reshape lại bên trong.
    """
    # Xác định số kênh (ví dụ: 3 cho RGB)
    # Giả sử kích thước cửa sổ là hình vuông, ví dụ 3x3 -> 9 pixels
    num_pixels = window_data.shape[0] // 3  # Chia cho 3 vì giả định là RGB
    window_size_sq = num_pixels

    # Reshape lại thành (num_pixels, num_channels)
    pixels = window_data.reshape((window_size_sq, 3))

    min_sum_dist = float('inf')
    median_vector = pixels[0]  # Khởi tạo tạm

    # Sử dụng tổng bình phương khoảng cách Euclidean để tránh căn bậc hai
    for i in range(window_size_sq):
        candidate = pixels[i]
        current_sum_sq_dist = 0
        for j in range(window_size_sq):
            diff = candidate - pixels[j]
            current_sum_sq_dist += np.dot(diff, diff)  # Bình phương khoảng cách L2

        if current_sum_sq_dist < min_sum_dist:
            min_sum_dist = current_sum_sq_dist
            median_vector = candidate

    # Chỉ trả về giá trị của pixel trung tâm tương ứng với vector trung vị
    # Tuy nhiên, generic_filter yêu cầu trả về một giá trị scalar.
    # Do đó, cách tiếp cận này không dùng generic_filter hiệu quả cho VMF.
    # Ta sẽ tự viết vòng lặp ngoài.
    # Hàm này sẽ được gọi trong vòng lặp thay vì qua generic_filter.
    return median_vector


def apply_vmf(image, window_size):
    """
    Áp dụng bộ lọc trung vị vector (VMF) lên toàn bộ ảnh.
    """
    height, width, channels = image.shape
    pad_size = (window_size - 1) // 2
    output_image = np.zeros_like(image, dtype=np.float64)

    # Thêm padding để xử lý biên
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

    print(f"Applying VMF with window {window_size}x{window_size}...")
    # Sử dụng tqdm để hiển thị thanh tiến trình
    for r in tqdm(range(height)):
        for c in range(width):
            # Trích xuất cửa sổ từ ảnh đã padding
            window = padded_image[r: r + window_size, c: c + window_size, :]
            # Reshape cửa sổ thành danh sách các vector pixel
            pixels_in_window = window.reshape(-1, channels)  # Shape: (window_size*window_size, channels)

            min_sum_sq_dist = float('inf')
            median_vector = pixels_in_window[0]

            # Tính toán vector trung vị cho cửa sổ hiện tại
            for i in range(pixels_in_window.shape[0]):
                candidate = pixels_in_window[i]
                current_sum_sq_dist = 0
                for j in range(pixels_in_window.shape[0]):
                    diff = candidate - pixels_in_window[j]
                    current_sum_sq_dist += np.dot(diff, diff)

                if current_sum_sq_dist < min_sum_sq_dist:
                    min_sum_sq_dist = current_sum_sq_dist
                    median_vector = candidate

            output_image[r, c, :] = median_vector

    return output_image


def adaptive_two_pass_vmf(noisy_image, W1_size=3, W2_size=3, a=1.0, b=1.0):
    """
    Thực hiện thuật toán Adaptive Two-Pass Vector Median Filter.

    Args:
        noisy_image (np.array): Ảnh màu đầu vào bị nhiễu (HxWxChannels, vd: RGB).
                                Giá trị pixel nên ở dạng float (0.0 - 255.0).
        W1_size (int): Kích thước cửa sổ cho lượt VMF đầu tiên.
        W2_size (int): Kích thước cửa sổ cho lượt VMF thứ hai.
        a (float): Tham số điều chỉnh ngưỡng phát hiện cột lỗi (alpha trong paper).
        b (float): Tham số điều chỉnh số lượng pixel thay thế (beta trong paper).

    Returns:
        np.array: Ảnh đã lọc nhiễu.
    """
    X = noisy_image.astype(np.float64)
    M, N, C = X.shape

    # --- Step 1: Lọc VMF lần 1 và tính chỉ số lỗi E1 ---
    print("Step 1: First VMF Pass")
    Y = apply_vmf(X, W1_size)

    # Tính toán ma trận lỗi E1
    # E1(m,n) = 1 nếu pixel bị thay đổi bởi VMF, ngược lại là 0
    # So sánh dựa trên khoảng cách vector (norm L2)
    difference_norm = np.linalg.norm(X - Y, axis=2)
    E1 = (difference_norm > 1e-6).astype(int)  # Sử dụng ngưỡng nhỏ để tránh lỗi số thực

    # # --- Step 2: Hiệu chỉnh thích ứng ---
    # print("Step 2: Adaptive Correction")
    # # Tính tỷ lệ nhiễu trung bình trên mỗi cột (lambda(n) trong paper)
    # lambda_ratio_per_column = np.sum(E1, axis=0) / M  # Shape (N,)
    #
    # # Tính tỷ lệ nhiễu trung bình toàn ảnh (Lambda trong paper - dùng mean thay vì sum/N)
    # Lambda_overall_ratio = np.mean(lambda_ratio_per_column)
    # # Hoặc tính trực tiếp: Lambda_overall_ratio = np.sum(E1) / (M * N)
    #
    # # Tính độ lệch chuẩn của tỷ lệ nhiễu cột
    # sigma_lambda = np.std(lambda_ratio_per_column)
    #
    # # Ngưỡng phát hiện cột lỗi
    # eta = a * sigma_lambda
    #
    # # Khởi tạo ảnh trung gian Y_tilde và ma trận lỗi E2
    # Y_tilde = Y.copy()
    # E2 = np.zeros_like(E1, dtype=int)
    #
    # print("Checking columns for over-correction...")
    # # Duyệt qua từng cột để kiểm tra và hiệu chỉnh
    # for n in tqdm(range(N)):
    #     # Kiểm tra nếu tỷ lệ nhiễu cột này cao bất thường
    #     if lambda_ratio_per_column[n] > Lambda_overall_ratio + eta:
    #         # Tính số lượng pixel cần khôi phục K (theo công thức paper)
    #         K = round((lambda_ratio_per_column[n] - Lambda_overall_ratio + b * sigma_lambda) * M)
    #         K = max(0, min(K, M))  # Đảm bảo K hợp lệ
    #
    #         if K > 0:
    #             # Tính vector chênh lệch và độ lớn của chúng trong cột n
    #             diff_vectors_col = X[:, n, :] - Y[:, n, :]  # Shape (M, C)
    #             diff_magnitudes_col = np.linalg.norm(diff_vectors_col, axis=1)  # Shape (M,)
    #
    #             # Tìm K chỉ số (hàng m) có độ lớn chênh lệch nhỏ nhất
    #             # Đây là những pixel ít có khả năng bị nhiễu nhất trong số những pixel đã bị thay đổi
    #             indices_to_revert = np.argsort(diff_magnitudes_col)[:K]
    #
    #             # Khôi phục giá trị gốc và đánh dấu vào E2
    #             for m in indices_to_revert:
    #                 # Chỉ khôi phục nếu pixel đó thực sự đã bị thay đổi ở bước 1
    #                 if E1[m, n] == 1:
    #                     Y_tilde[m, n, :] = X[m, n, :]
    #                     E2[m, n] = 1

    # --- Step 2: Adaptive Correction (Cách tính khác) ---
    print("Step 2: Adaptive Correction (Alternative Lambda Calculation)")
    # Tính *số lượng* pixel lỗi trên mỗi cột
    lambda_count_per_column = np.sum(E1, axis=0)  # Shape (N,)

    # Tính *số lượng* pixel lỗi trung bình trên mỗi cột (theo Eq 5 paper)
    Lambda_avg_count_per_column = np.sum(E1) / N  # Giá trị vô hướng

    # Tính độ lệch chuẩn của *số lượng* pixel lỗi cột
    sigma_lambda_count = np.std(lambda_count_per_column)

    # Ngưỡng phát hiện cột lỗi (dựa trên số lượng)
    eta_count = a * sigma_lambda_count

    Y_tilde = Y.copy()
    E2 = np.zeros_like(E1, dtype=int)

    print("Checking columns for over-correction (using counts)...")
    for n in tqdm(range(N)):
        # Kiểm tra nếu *số lượng* nhiễu cột này cao bất thường
        if lambda_count_per_column[n] > Lambda_avg_count_per_column + eta_count:  # So sánh số lượng
            # Tính K (cần điều chỉnh công thức cho phù hợp với số lượng thay vì tỷ lệ)
            # Ước lượng số pixel "dư thừa" so với trung bình cộng với độ lệch chuẩn
            excess_pixels = lambda_count_per_column[n] - Lambda_avg_count_per_column
            # Có thể dùng K = round(excess_pixels + b * sigma_lambda_count) hoặc một cách khác
            K = round(excess_pixels + b * sigma_lambda_count * (
                        lambda_count_per_column[n] / M))  # heuristic điều chỉnh theo tỷ lệ cột
            K = max(0, min(K, int(lambda_count_per_column[n])))  # K không vượt quá số pixel lỗi trong cột

            if K > 0:
                # ... (phần tìm diff_magnitudes_col và indices_to_revert giữ nguyên) ...
                diff_vectors_col = X[:, n, :] - Y[:, n, :]
                diff_magnitudes_col = np.linalg.norm(diff_vectors_col, axis=1)

                # Lấy chỉ số của các pixel đã bị thay đổi trong cột này (E1=1)
                changed_indices_in_col = np.where(E1[:, n] == 1)[0]

                if len(changed_indices_in_col) > 0:
                    # Chỉ xét các pixel đã bị thay đổi để khôi phục
                    magnitudes_of_changed = diff_magnitudes_col[changed_indices_in_col]

                    # Sắp xếp các pixel đã thay đổi theo độ lớn chênh lệch tăng dần
                    sorted_relative_indices = np.argsort(magnitudes_of_changed)

                    # Chọn K pixel đầu tiên từ danh sách đã sắp xếp này
                    num_to_revert = min(K, len(sorted_relative_indices))
                    absolute_indices_to_revert = changed_indices_in_col[sorted_relative_indices[:num_to_revert]]

                    # Khôi phục giá trị gốc và đánh dấu vào E2
                    for m in absolute_indices_to_revert:
                        Y_tilde[m, n, :] = X[m, n, :]
                        E2[m, n] = 1
    # --- Kết thúc Step 2 ---

    # --- Step 3: Lọc VMF lần 2 với hiệu chỉnh ---
    print("Step 3: Second VMF Pass with correction")
    # Áp dụng VMF lần 2 lên ảnh đã hiệu chỉnh Y_tilde
    Z_filtered = apply_vmf(Y_tilde, W2_size)

    # Tạo ảnh kết quả cuối cùng Z
    Z = Z_filtered.copy()

    # Đảm bảo những pixel đã được khôi phục ở Bước 2 giữ nguyên giá trị gốc X
    # Tìm chỉ số nơi E2 = 1
    revert_indices = np.where(E2 == 1)
    # Copy giá trị từ X sang Z tại các vị trí đó
    Z[revert_indices[0], revert_indices[1], :] = X[revert_indices[0], revert_indices[1], :]

    # Chuyển về kiểu dữ liệu gốc (ví dụ uint8) và clip giá trị
    Z = np.clip(Z, 0, 255).astype(np.uint8)

    return Z


# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    # Đọc ảnh màu (ví dụ: sử dụng OpenCV)
    input_image_path = 'images.jpg'  # Thay bằng đường dẫn ảnh nhiễu của bạn
    # Giả sử bạn đã có ảnh nhiễu 'noisy_color_image.png'
    # Nếu chưa có, bạn có thể tạo nhiễu Salt & Pepper cho ảnh màu
    # original_image = cv2.imread('lena_color.png') # Ví dụ đọc ảnh gốc
    # if original_image is None:
    #     print(f"Không thể đọc ảnh gốc!")
    # else:
    #     # Thêm nhiễu Salt & Pepper (ví dụ)
    #     noise_ratio = 0.1 # 10% nhiễu
    #     noisy_image_sp = original_image.copy()
    #     num_noise = int(noise_ratio * original_image.size / original_image.shape[2]) # Chia cho số kênh

    #     # Salt noise
    #     coords_salt = [np.random.randint(0, i - 1, num_noise // 2) for i in original_image.shape[:2]]
    #     noisy_image_sp[coords_salt[0], coords_salt[1], :] = [255, 255, 255]

    #     # Pepper noise
    #     coords_pepper = [np.random.randint(0, i - 1, num_noise // 2) for i in original_image.shape[:2]]
    #     noisy_image_sp[coords_pepper[0], coords_pepper[1], :] = [0, 0, 0]

    #     cv2.imwrite(input_image_path, noisy_image_sp)
    #     print(f"Đã tạo ảnh nhiễu: {input_image_path}")

    noisy_image = cv2.imread(input_image_path)

    if noisy_image is None:
        print(f"Lỗi: Không thể đọc ảnh nhiễu tại '{input_image_path}'")
        print("Hãy đảm bảo bạn có file ảnh nhiễu hoặc bỏ comment phần tạo nhiễu ở trên.")
    else:
        print(f"Đã đọc ảnh nhiễu: {input_image_path}, kích thước: {noisy_image.shape}")

        # Chạy thuật toán lọc
        # Chuyển sang float để tính toán
        filtered_image = adaptive_two_pass_vmf(noisy_image.astype(np.float64),
                                               W1_size=3,
                                               W2_size=3,
                                               a=1.0, #giảm a để chạy hơn
                                               b=1.0) #tăng b để khôi phục nhiều hơn

        # Lưu ảnh kết quả
        output_image_path = 'filtered_adaptive_vmf.png'
        cv2.imwrite(output_image_path, filtered_image)
        print(f"Đã lọc và lưu ảnh kết quả: {output_image_path}")

        # Hiển thị ảnh (tùy chọn)
        # cv2.imshow('Noisy Image', noisy_image)
        # cv2.imshow('Filtered Image (Adaptive VMF)', filtered_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()