import numpy as np
from scipy.ndimage import median_filter
import cv2 # Hoặc from PIL import Image
import matplotlib.pyplot as plt # Thư viện để vẽ biểu đồ
import os # Để kiểm tra file tồn tại

def adaptive_two_pass_median_filter(noisy_image, w1_size=3, w2_size=3, a=1.0, b=1.0):
    """
    Thực hiện Bộ lọc trung vị thích ứng hai lượt.
    (Giữ nguyên hàm này như bạn đã cung cấp)
    """
    if noisy_image.ndim != 2:
        # Nếu ảnh màu, thử chuyển sang thang xám
        if noisy_image.ndim == 3 and noisy_image.shape[2] in [3, 4]:
            print("Cảnh báo: Ảnh đầu vào là ảnh màu, đang chuyển sang thang xám.")
            gray_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
            X = gray_image.astype(float)
        else:
            raise ValueError("Hàm này chỉ hỗ trợ ảnh thang xám (2 chiều) hoặc có thể chuyển đổi.")
    else:
         X = noisy_image.astype(float) # Làm việc với float để tránh lỗi tràn số

    # Đảm bảo ảnh gốc để tính toán là thang xám
    if noisy_image.ndim == 3:
      reference_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY).astype(float)
    else:
      reference_image = noisy_image.astype(float)


    M, N = X.shape

    # --- Bước 1: Lọc trung vị lần đầu ---
    Y = median_filter(X, size=w1_size, mode='reflect') # mode='reflect' xử lý biên ảnh

    # Tính ma trận lỗi E1
    diff_xy = reference_image - Y # So sánh Y với ảnh gốc thang xám
    E1 = (diff_xy != 0).astype(int)

    # --- Bước 2: Thay thế thích ứng ---
    lambda_n = np.mean(E1, axis=0)
    mean_lambda = np.mean(lambda_n)
    sigma_lambda = np.std(lambda_n)
    eta = a * sigma_lambda

    Y_tilde = Y.copy()
    E2 = np.zeros_like(X, dtype=int)

    for n in range(N):
        if (lambda_n[n] - mean_lambda) > eta:
            K_float = M * (lambda_n[n] - mean_lambda + b * sigma_lambda)
            K = int(round(K_float))
            K = max(0, min(M, K))

            if K > 0:
                e_col = reference_image[:, n] - Y[:, n] # So sánh Y với ảnh gốc thang xám
                abs_e_col = np.abs(e_col)
                indices_in_col = np.where(E1[:, n] == 1)[0]

                if len(indices_in_col) > 0:
                   abs_e_at_indices = abs_e_col[indices_in_col]
                   sorted_indices = indices_in_col[np.argsort(abs_e_at_indices)]
                   v = sorted_indices[:K]

                   if len(v) > 0:
                       # Quan trọng: Chỉ khôi phục giá trị gốc thang xám
                       Y_tilde[v, n] = reference_image[v, n]
                       E2[v, n] = 1

    # --- Bước 3: Lọc trung vị lần hai ---
    Z_temp = median_filter(Y_tilde, size=w2_size, mode='reflect')
    Z = np.where(E2 == 1, Y_tilde, Z_temp)

    # Chuyển về kiểu dữ liệu gốc nếu cần (ví dụ uint8 cho ảnh)
    # Xác định kiểu dữ liệu gốc từ noisy_image ban đầu
    original_dtype = noisy_image.dtype
    if np.issubdtype(original_dtype, np.integer):
        # Giả sử giới hạn 0-255 cho kiểu integer (phổ biến nhất là uint8)
        max_val = np.iinfo(original_dtype).max if np.iinfo(original_dtype).max <= 255 else 255
        min_val = np.iinfo(original_dtype).min if np.iinfo(original_dtype).min >= 0 else 0
        Z = np.clip(Z, min_val, max_val).astype(original_dtype)
    elif np.issubdtype(original_dtype, np.floating):
         Z = Z.astype(original_dtype)
    else: # Fallback nếu không phải integer hay float quen thuộc
        Z = np.clip(Z, 0, 255).astype(np.uint8)


    return Z

# --- Các hàm thêm nhiễu ---

def add_salt_and_pepper_noise(image, percentage):
    """Thêm nhiễu Salt and Pepper."""
    noisy_image = image.copy()
    rows, cols = image.shape[:2] # Hoạt động với cả ảnh màu và xám
    num_noise_pixels = int(rows * cols * percentage)

    x_coords = np.random.randint(0, rows, num_noise_pixels)
    y_coords = np.random.randint(0, cols, num_noise_pixels)

    num_salt = num_noise_pixels // 2
    num_pepper = num_noise_pixels - num_salt

    # Xác định giá trị min/max dựa trên dtype
    if np.issubdtype(image.dtype, np.integer):
      max_val = np.iinfo(image.dtype).max
      min_val = np.iinfo(image.dtype).min
    else: # Giả sử float trong [0, 1] hoặc [0, 255] - dùng 0, 255 cho đơn giản
      max_val = 255
      min_val = 0


    # Thêm nhiễu salt
    noisy_image[x_coords[:num_salt], y_coords[:num_salt]] = max_val
    # Thêm nhiễu pepper
    noisy_image[x_coords[num_salt:], y_coords[num_salt:]] = min_val

    return noisy_image

def add_gaussian_noise(image, std_dev):
    """Thêm nhiễu Gaussian."""
    # Chuyển sang float để cộng nhiễu chính xác
    img_float = image.astype(np.float64)
    mean = 0
    # Tạo nhiễu Gaussian
    gaussian_noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image_float = img_float + gaussian_noise

    # Clip và chuyển về kiểu dữ liệu gốc
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        min_val = np.iinfo(image.dtype).min
        noisy_image = np.clip(noisy_image_float, min_val, max_val).astype(image.dtype)
    else: # Giả sử float [0, 1] hoặc [0, 255]
         # Cần xác định rõ phạm vi ảnh float gốc, ở đây giả định là 0-255
         noisy_image = np.clip(noisy_image_float, 0, 255).astype(image.dtype)

    return noisy_image

def add_speckle_noise(image, variance):
    """Thêm nhiễu Speckle (multiplicative noise)."""
     # Chuyển sang float để nhân nhiễu chính xác
    img_float = image.astype(np.float64)
    mean = 0
    std_dev = variance**0.5
    # Tạo nhiễu Gaussian cho thành phần nhân
    # Kích thước phải khớp với ảnh (quan trọng nếu ảnh màu)
    speckle_noise = np.random.normal(mean, std_dev, image.shape)

    # Nhiễu speckle là nhiễu nhân
    noisy_image_float = img_float + img_float * speckle_noise

    # Clip và chuyển về kiểu dữ liệu gốc
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        min_val = np.iinfo(image.dtype).min
        noisy_image = np.clip(noisy_image_float, min_val, max_val).astype(image.dtype)
    else: # Giả sử float [0, 1] hoặc [0, 255]
         # Cần xác định rõ phạm vi ảnh float gốc, ở đây giả định là 0-255
         noisy_image = np.clip(noisy_image_float, 0, 255).astype(image.dtype)

    return noisy_image


# --- Các hàm tính lỗi ---
def calculate_mse(image_original, image_processed):
    """Tính Mean Squared Error (MSE) giữa hai ảnh."""
    if image_original.shape != image_processed.shape:
        # Nếu kích thước khác nhau (ví dụ: ảnh gốc màu, ảnh xử lý xám), chuyển ảnh gốc sang xám
        if image_original.ndim == 3 and image_processed.ndim == 2:
             img_orig_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        elif image_original.ndim == 2 and image_processed.ndim == 3:
             # Trường hợp này ít xảy ra với bộ lọc này
             print("Cảnh báo: Ảnh gốc thang xám, ảnh xử lý có vẻ là màu?")
             # Thử chuyển ảnh xử lý sang xám (giả định)
             img_proc_gray = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
             image_processed = img_proc_gray # Cập nhật để tính toán
             img_orig_gray = image_original
        else:
            raise ValueError(f"Kích thước hai ảnh không khớp và không thể tự động chuyển đổi: {image_original.shape} vs {image_processed.shape}")
    else:
        # Nếu cả hai cùng màu hoặc cùng xám
        if image_original.ndim == 3:
             img_orig_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
             # Bộ lọc trả về ảnh xám, nên không cần chuyển image_processed
             if image_processed.ndim == 3: # Đề phòng trường hợp bất thường
                  image_processed = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)

        else:
             img_orig_gray = image_original # Cả hai đã là thang xám

    if img_orig_gray.shape != image_processed.shape:
         raise ValueError(f"Kích thước ảnh không khớp sau khi chuyển đổi: {img_orig_gray.shape} vs {image_processed.shape}")

    # Tính toán trên ảnh thang xám
    err = np.sum((img_orig_gray.astype("float") - image_processed.astype("float")) ** 2)
    mse = err / float(img_orig_gray.size) # Dùng size để đúng với cả ảnh màu/xám gốc
    return mse

def calculate_mae(image_original, image_processed):
    """Tính Mean Absolute Error (MAE) giữa hai ảnh."""
    # Xử lý kích thước và chuyển đổi sang thang xám tương tự MSE
    if image_original.shape != image_processed.shape:
        if image_original.ndim == 3 and image_processed.ndim == 2:
             img_orig_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        elif image_original.ndim == 2 and image_processed.ndim == 3:
             print("Cảnh báo: Ảnh gốc thang xám, ảnh xử lý có vẻ là màu?")
             img_proc_gray = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
             image_processed = img_proc_gray
             img_orig_gray = image_original
        else:
            raise ValueError(f"Kích thước hai ảnh không khớp và không thể tự động chuyển đổi: {image_original.shape} vs {image_processed.shape}")
    else:
        if image_original.ndim == 3:
             img_orig_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
             if image_processed.ndim == 3:
                  image_processed = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
        else:
             img_orig_gray = image_original

    if img_orig_gray.shape != image_processed.shape:
         raise ValueError(f"Kích thước ảnh không khớp sau khi chuyển đổi: {img_orig_gray.shape} vs {image_processed.shape}")

    # Tính toán trên ảnh thang xám
    err = np.sum(np.abs(img_orig_gray.astype("float") - image_processed.astype("float")))
    mae = err / float(img_orig_gray.size)
    return mae

# --- Chương trình chính ---
if __name__ == '__main__':
    # --- 1. Tải ảnh gốc ---
    original_image_path = 'lena.jpg' # Giữ nguyên hoặc đổi tên file ảnh của bạn
    if not os.path.exists(original_image_path):
        print(f"Lỗi: Không tìm thấy file ảnh gốc tại '{original_image_path}'")
        print("Vui lòng đảm bảo file ảnh tồn tại trong cùng thư mục hoặc cung cấp đường dẫn đầy đủ.")
        # Tùy chọn: Tải ảnh mặc định nếu không tìm thấy
        try:
            # Cố gắng tải ảnh test mặc định của OpenCV nếu có
            import cv2.data
            lena_path_default = os.path.join(cv2.data.haarcascades, '../lena.jpg')
            if os.path.exists(lena_path_default):
                 print(f"Đang thử tải ảnh mặc định: {lena_path_default}")
                 original_image_path = lena_path_default
            else:
                 # Nếu vẫn không có, tạo ảnh giả để chạy code (ít ý nghĩa)
                 print("Không tìm thấy ảnh gốc và ảnh mặc định. Tạo ảnh xám ngẫu nhiên.")
                 img_original = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
                 # Gán lại path để thông báo khớp
                 original_image_path = "Ảnh ngẫu nhiên"
                 # Không cần đọc lại từ file nếu đã tạo
                 img_read_success = True
                 if img_original is None: img_read_success = False # Kiểm tra lại
            # Nếu không phải tạo ảnh giả, đọc lại ảnh mặc định
            if original_image_path != "Ảnh ngẫu nhiên":
                 img_original = cv2.imread(original_image_path) # Đọc ảnh màu nếu có
                 img_read_success = img_original is not None
        except Exception as e_load:
            print(f"Không thể tải ảnh mặc định hoặc tạo ảnh giả: {e_load}")
            exit()
    else:
        # Đọc ảnh gốc (có thể là màu hoặc xám)
        img_original = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
        img_read_success = img_original is not None


    if not img_read_success:
        print(f"Không thể đọc file ảnh: {original_image_path}")
        exit()

    print(f"Đã tải ảnh gốc thành công: {original_image_path} - Kích thước: {img_original.shape}, Kiểu dữ liệu: {img_original.dtype}")

    # Luôn chuyển ảnh gốc sang thang xám để làm tham chiếu chuẩn cho tính lỗi
    if img_original.ndim == 3:
        img_original_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    else:
        img_original_gray = img_original.copy() # Đã là thang xám

    # --- 2. Thiết lập các mức nhiễu và loại nhiễu ---
    noise_percentages = np.arange(5, 101, 5) # Từ 5% đến 100%, bước nhảy 5%
    noise_types = ['salt_pepper', 'gaussian', 'speckle']

    # Lưu kết quả MAE cho từng loại nhiễu
    mae_results = {noise: [] for noise in noise_types}
    # (Tùy chọn) Lưu cả MSE
    # mse_results = {noise: [] for noise in noise_types}

    # Ánh xạ % sang tham số nhiễu Gaussian và Speckle (có thể cần điều chỉnh)
    # Chọn giá trị sao cho mức nhiễu cao nhất (100%) là đáng kể
    MAX_GAUSSIAN_STD_DEV = 50.0
    MAX_SPECKLE_VARIANCE = 0.5 # Variance càng lớn nhiễu càng mạnh

    print("\nBắt đầu quá trình thêm nhiễu, lọc và tính lỗi cho các loại nhiễu...")

    # --- 3. Vòng lặp qua các loại nhiễu và mức nhiễu ---
    for noise_type in noise_types:
        print(f"\n--- Xử lý loại nhiễu: {noise_type.replace('_', ' ').title()} ---")
        for percent in noise_percentages:
            noise_level = percent / 100.0
            print(f"  Đang xử lý mức {percent}%...")

            # Thêm nhiễu vào ảnh gốc (có thể là màu hoặc xám tùy file gốc)
            if noise_type == 'salt_pepper':
                img_noisy = add_salt_and_pepper_noise(img_original, noise_level)
            elif noise_type == 'gaussian':
                # Ánh xạ % sang std_dev
                current_std_dev = noise_level * MAX_GAUSSIAN_STD_DEV
                img_noisy = add_gaussian_noise(img_original, current_std_dev)
            elif noise_type == 'speckle':
                # Ánh xạ % sang variance
                current_variance = noise_level * MAX_SPECKLE_VARIANCE
                img_noisy = add_speckle_noise(img_original, current_variance)

            # Áp dụng bộ lọc thích ứng (luôn xử lý trên thang xám)
            # Hàm adaptive_two_pass_median_filter đã xử lý việc chuyển sang xám nếu cần
            img_filtered = adaptive_two_pass_median_filter(img_noisy, w1_size=3, w2_size=3, a=1.0, b=1.0)

            # Tính toán MAE (và MSE nếu muốn) giữa ảnh gốc THANG XÁM và ảnh đã lọc
            # Hàm calculate_mae đã xử lý việc đảm bảo so sánh trên thang xám
            current_mae = calculate_mae(img_original_gray, img_filtered) # Luôn dùng ảnh gốc xám
            mae_results[noise_type].append(current_mae)

            # (Tùy chọn) Tính MSE
            # current_mse = calculate_mse(img_original_gray, img_filtered)
            # mse_results[noise_type].append(current_mse)

            # (Tùy chọn) Hiển thị ảnh ở một mức nhiễu nào đó để kiểm tra
            # if percent == 50:
            #     cv2.imshow(f'Anh goc (Gray)', img_original_gray)
            #     # Hiển thị ảnh nhiễu (có thể màu hoặc xám)
            #     cv2.imshow(f'Nhieu {noise_type} {percent}%', img_noisy)
            #     # Ảnh lọc luôn là thang xám
            #     cv2.imshow(f'Loc voi nhieu {noise_type} {percent}%', img_filtered)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

    print("\nHoàn tất quá trình tính toán lỗi.")

    # --- 4. Vẽ biểu đồ so sánh MAE ---
    print("Đang vẽ biểu đồ so sánh MAE...")
    plt.figure(figsize=(10, 6)) # Kích thước cửa sổ biểu đồ

    colors = {'salt_pepper': 'r', 'gaussian': 'g', 'speckle': 'm'}
    markers = {'salt_pepper': 's', 'gaussian': 'o', 'speckle': '^'}
    linestyles = {'salt_pepper': '--', 'gaussian': '-', 'speckle': ':'}
    labels = {'salt_pepper': 'Salt & Pepper', 'gaussian': 'Gaussian', 'speckle': 'Speckle'}

    for noise_type in noise_types:
        plt.plot(noise_percentages, mae_results[noise_type],
                 marker=markers[noise_type],
                 linestyle=linestyles[noise_type],
                 color=colors[noise_type],
                 label=labels[noise_type])

    plt.title('So sánh MAE của bộ lọc thích ứng với các loại nhiễu')
    plt.xlabel('Mức độ nhiễu (%)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xticks(noise_percentages[::2]) # Hiển thị các mốc % cách nhau 10% cho đỡ rối
    plt.legend() # Hiển thị chú giải cho các đường
    plt.grid(True) # Hiện lưới
    plt.tight_layout() # Tự động điều chỉnh
    plt.show() # Hiển thị cửa sổ biểu đồ

    # (Tùy chọn) Vẽ biểu đồ MSE nếu bạn đã tính
    # print("Đang vẽ biểu đồ so sánh MSE...")
    # plt.figure(figsize=(10, 6))
    # for noise_type in noise_types:
    #     plt.plot(noise_percentages, mse_results[noise_type],
    #              marker=markers[noise_type],
    #              linestyle=linestyles[noise_type],
    #              color=colors[noise_type],
    #              label=labels[noise_type])
    # plt.title('So sánh MSE của bộ lọc thích ứng với các loại nhiễu')
    # plt.xlabel('Mức độ nhiễu (%)')
    # plt.ylabel('Mean Squared Error (MSE)')
    # plt.xticks(noise_percentages[::2])
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    print("\nHoàn thành!")