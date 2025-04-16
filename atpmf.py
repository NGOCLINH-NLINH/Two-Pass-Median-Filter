import numpy as np
from scipy.ndimage import median_filter
import cv2 # Sử dụng OpenCV để đọc/ghi ảnh (tùy chọn)

def atpmf_grayscale(X, W1_size=3, W2_size=3, a=1.0, b=1.0):
    """
    Thực hiện bộ lọc trung vị hai lượt thích ứng cho ảnh thang độ xám.

    Tham số:
        X (numpy.ndarray): Ảnh thang độ xám đầu vào (dạng float, giá trị 0-255 hoặc 0-1).
        W1_size (int): Kích thước cửa sổ lọc trung vị lần 1 (ví dụ: 3 cho 3x3).
        W2_size (int): Kích thước cửa sổ lọc trung vị lần 2.
        a (float): Tham số điều khiển ngưỡng phát hiện cột 'sửa lỗi quá mức'.
        b (float): Tham số điều khiển số lượng pixel được khôi phục.

    Trả về:
        numpy.ndarray: Ảnh thang độ xám đã được lọc.
    """
    if X.ndim != 2:
        raise ValueError("Ảnh đầu vào phải là ảnh thang độ xám (2 chiều).")
    if W1_size % 2 == 0 or W2_size % 2 == 0:
        raise ValueError("Kích thước cửa sổ lọc phải là số lẻ.")

    # Đảm bảo ảnh ở dạng float để tính toán chính xác
    X_float = X.astype(np.float64)
    M, N = X_float.shape

    # --- Bước 1: Lọc trung vị lần 1 ---
    Y = median_filter(X_float, size=W1_size, mode='reflect')
    # Ma trận chỉ số lỗi 1 (1 nếu X khác Y, 0 nếu giống)
    E1 = (np.abs(X_float - Y) > 1e-6).astype(int) # So sánh float với sai số nhỏ

    # --- Bước 2: Bước Thích ứng ---
    Y_tilde = Y.copy() # Ảnh trung gian để khôi phục
    E2 = np.zeros_like(X_float, dtype=int) # Ma trận chỉ số lỗi 2

    # Tính toán thống kê nhiễu
    num_noisy_pixels_total = np.sum(E1)
    if M * N > 0:
         # Tỷ lệ nhiễu trung bình ước tính toàn ảnh (theo paper là chia N, nhưng hợp lý hơn là M*N)
         # Chúng ta sẽ dùng Lambda là tỷ lệ lỗi ước tính trên mỗi cột trung bình
        lambda_n = np.sum(E1, axis=0) / M if M > 0 else np.zeros(N) # Tỷ lệ nhiễu trên từng cột
        Lambda = np.mean(lambda_n) # Lấy trung bình tỷ lệ lỗi các cột
    else:
        Lambda = 0
        lambda_n = np.zeros(N)

    if N > 1:
      sigma_lambda = np.std(lambda_n)
    else:
      sigma_lambda = 0 # Tránh lỗi nếu chỉ có 1 cột

    # Ngưỡng phát hiện cột "sửa lỗi quá mức"
    eta = a * sigma_lambda
    epsilon = 1e-9 # Ngưỡng nhỏ để kiểm tra sigma_lambda

    # Xử lý từng cột
    for n in range(N):
        # Kiểm định giả thuyết (một phía)
        # Chỉ thực hiện nếu sigma_lambda đáng kể và tỷ lệ cột vượt ngưỡng Lambda + eta
        if sigma_lambda > epsilon and (lambda_n[n] - Lambda) > eta:
            # Cột này bị nghi ngờ "sửa lỗi quá mức"
            e_col = X_float[:, n] - Y[:, n] # Sai khác giữa gốc và kết quả lọc 1

            # Xác định số pixel cần khôi phục (K)
            # Dựa theo công thức Fig 4, nhưng nhân với M để ra số pixel
            # Đảm bảo K không âm
            K = int(round(max(0.0, (lambda_n[n] - Lambda + b * sigma_lambda)) * M))
            # K = int(round(max(0.0, (lambda_n[n] - (Lambda + eta) + b * sigma_lambda)) * M)) # Cách diễn giải khác

            if K > 0:
                # Tìm chỉ số của các pixel đã bị thay đổi trong cột này (E1=1)
                changed_indices = np.where(E1[:, n] == 1)[0]

                if len(changed_indices) > 0:
                    # Tính sai số tuyệt đối tại các vị trí bị thay đổi
                    abs_errors_at_changed = np.abs(e_col[changed_indices])

                    # Số lượng pixel tối đa có thể khôi phục là số pixel đã thay đổi
                    num_to_restore = min(K, len(changed_indices))

                    if num_to_restore > 0:
                        # Tìm chỉ số (trong mảng changed_indices) của K lỗi nhỏ nhất
                        # argsort trả về chỉ số để sắp xếp mảng theo thứ tự tăng dần
                        sorted_local_indices = np.argsort(abs_errors_at_changed)
                        indices_to_restore_local = sorted_local_indices[:num_to_restore]

                        # Chuyển về chỉ số gốc trong cột (indices trong mảng 0 đến M-1)
                        indices_to_restore_global = changed_indices[indices_to_restore_local]

                        # Khôi phục giá trị gốc và đánh dấu E2
                        Y_tilde[indices_to_restore_global, n] = X_float[indices_to_restore_global, n]
                        E2[indices_to_restore_global, n] = 1

    # --- Bước 3: Lọc trung vị lần 2 (có điều kiện) ---
    # Chỉ lọc những pixel không được đánh dấu trong E2 (nghĩa là không được khôi phục)
    # Cách 1: Lọc toàn bộ rồi kết hợp (như trong mô tả trước)
    Z_pass2 = median_filter(Y_tilde, size=W2_size, mode='reflect')
    Z = np.where(E2 == 1, Y_tilde, Z_pass2) # Giữ nguyên pixel đã khôi phục (E2=1), còn lại lấy kết quả lọc 2

    # # Cách 2: Tạo mask và chỉ lọc các pixel không bị mask (phức tạp hơn với median_filter của scipy)
    # # Cần triển khai median filter tùy chỉnh nếu muốn làm theo cách này hiệu quả.

    # Chuyển về kiểu dữ liệu gốc nếu cần (ví dụ uint8)
    # if X.dtype == np.uint8:
    #     Z = np.clip(Z, 0, 255).astype(np.uint8)
    # else:
    #     # Giữ nguyên float nếu đầu vào là float
    #     pass # Hoặc clip về [0, 1] nếu đầu vào là float 0-1

    return Z

# --- Hàm xử lý ảnh màu bằng cách áp dụng lên từng kênh ---
def atpmf_color_per_channel(X_color, W1_size=3, W2_size=3, a=1.0, b=1.0):
    """
    Áp dụng bộ lọc ATPMF lên từng kênh màu của ảnh một cách độc lập.

    Tham số:
        X_color (numpy.ndarray): Ảnh màu đầu vào (ví dụ: định dạng BGR của OpenCV hoặc RGB).
                                   Giả định là 3 kênh màu.
        W1_size (int): Kích thước cửa sổ lọc trung vị lần 1.
        W2_size (int): Kích thước cửa sổ lọc trung vị lần 2.
        a (float): Tham số 'a' cho bước thích ứng.
        b (float): Tham số 'b' cho bước thích ứng.

    Trả về:
        numpy.ndarray: Ảnh màu đã được lọc.
    """
    if X_color.ndim != 3 or X_color.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải là ảnh màu 3 kênh.")

    # Lưu kiểu dữ liệu gốc
    original_dtype = X_color.dtype
    # Chuyển sang float để tính toán
    X_color_float = X_color.astype(np.float64)

    # Tách các kênh màu
    channels = cv2.split(X_color_float) # Hoặc channels = [X_color_float[:,:,i] for i in range(3)]

    processed_channels = []
    for channel in channels:
        # Áp dụng bộ lọc ATPMF cho từng kênh
        processed_channel = atpmf_grayscale(channel, W1_size, W2_size, a, b)
        processed_channels.append(processed_channel)

    # Ghép các kênh đã xử lý lại
    Z_color = cv2.merge(processed_channels)

    # Chuyển về kiểu dữ liệu gốc
    if original_dtype == np.uint8:
         # Clip giá trị về khoảng 0-255 trước khi chuyển kiểu uint8
        Z_color = np.clip(Z_color, 0, 255).astype(np.uint8)
    # else: giữ nguyên float nếu đầu vào là float (ví dụ 0-1)
    #     Z_color = np.clip(Z_color, 0, 1.0) # Nếu đầu vào là float 0-1

    return Z_color

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    # --- Ví dụ cho ảnh thang độ xám ---
    try:
        # Đọc ảnh thang độ xám (thay 'grayscale_image.png' bằng đường dẫn thực tế)
        gray_image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            print("Không thể đọc ảnh thang độ xám.")
        else:
            print("Đã đọc ảnh thang độ xám.")

            # Hiển thị ảnh gốc
            cv2.imshow('original.png', gray_image)

            # Thêm nhiễu xung (ví dụ: salt & pepper) để kiểm tra
            noise_ratio = 0.5 # 20% nhiễu
            noisy_gray_image = gray_image.copy().astype(np.float64)
            num_noise = int(noise_ratio * gray_image.size)

            # Thêm nhiễu salt (trắng)
            salt_coords = [np.random.randint(0, i - 1, num_noise // 2) for i in gray_image.shape]
            noisy_gray_image[salt_coords[0], salt_coords[1]] = 255

            # Thêm nhiễu pepper (đen)
            pepper_coords = [np.random.randint(0, i - 1, num_noise // 2) for i in gray_image.shape]
            noisy_gray_image[pepper_coords[0], pepper_coords[1]] = 0

            noisy_gray_image = noisy_gray_image.astype(gray_image.dtype)
            cv2.imshow('noisy_gray_image.png', noisy_gray_image)
            print("Đã tạo ảnh thang độ xám bị nhiễu: noisy_gray_image.png")

            # Áp dụng bộ lọc ATPMF
            filtered_gray_image = atpmf_grayscale(noisy_gray_image, W1_size=3, W2_size=3, a=1.0, b=1.0)
            cv2.imshow('filtered_atpmf_gray.png', filtered_gray_image)
            print("Đã lọc ảnh thang độ xám bằng ATPMF: filtered_atpmf_gray.png")

            # So sánh với Median Filter thông thường
            filtered_median_gray = cv2.medianBlur(noisy_gray_image, 3)
            cv2.imshow('filtered_median_gray.png', filtered_median_gray)
            print("Đã lọc ảnh thang độ xám bằng Median thông thường: filtered_median_gray.png")

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy tệp ảnh thang độ xám mẫu.")
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh thang độ xám: {e}")


    # --- Ví dụ cho ảnh màu ---
    try:
        # Đọc ảnh màu (thay 'color_image.png' bằng đường dẫn thực tế)
        color_image = cv2.imread('images.jpg', cv2.IMREAD_COLOR)
        if color_image is None:
            print("Không thể đọc ảnh màu.")
        else:
            print("Đã đọc ảnh màu.")

            # Hiển thị ảnh gốc
            cv2.imshow('original_color_image.png', color_image)

            # Thêm nhiễu xung cho ảnh màu (ví dụ salt & pepper trên từng kênh)
            noise_ratio = 1
            noisy_color_image = color_image.copy()
            num_noise_per_channel = int(noise_ratio * color_image.shape[0] * color_image.shape[1]) // 3

            for i in range(3): # Lặp qua 3 kênh màu
                # Salt noise
                coords = [np.random.randint(0, dim - 1, num_noise_per_channel // 2) for dim in color_image.shape[:2]]
                noisy_color_image[coords[0], coords[1], i] = 255
                 # Pepper noise
                coords = [np.random.randint(0, dim - 1, num_noise_per_channel // 2) for dim in color_image.shape[:2]]
                noisy_color_image[coords[0], coords[1], i] = 0

            cv2.imshow('noisy_color_image.png', noisy_color_image)
            print("Đã tạo ảnh màu bị nhiễu: noisy_color_image.png")

             # Áp dụng bộ lọc ATPMF trên từng kênh
            # filtered_color_image = atpmf_color_per_channel(noisy_color_image, W1_size=3, W2_size=3, a=1.0, b=1.0)
            filtered_color_image = atpmf_color_per_channel(color_image, W1_size=3, W2_size=3, a=1.0, b=1.0)

            cv2.imshow('filtered_atpmf_color.png', filtered_color_image)
            print("Đã lọc ảnh màu bằng ATPMF (từng kênh): filtered_atpmf_color.png")

            # So sánh với Median Filter thông thường (áp dụng từng kênh không tối ưu màu)
            # Hoặc dùng medianBlur của OpenCV (thường hiệu quả hơn cho màu)
            filtered_median_color = cv2.medianBlur(noisy_color_image, 3)
            cv2.imshow('filtered_median_color.png', filtered_median_color)
            print("Đã lọc ảnh màu bằng Median thông thường (cv2.medianBlur): filtered_median_color.png")


            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except FileNotFoundError:
         print("Lỗi: Không tìm thấy tệp ảnh màu mẫu.")
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh màu: {e}")