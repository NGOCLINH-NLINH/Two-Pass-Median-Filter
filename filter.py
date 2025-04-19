import numpy as np
from scipy.ndimage import median_filter
import cv2 # Hoặc from PIL import Image
import matplotlib.pyplot as plt # Thư viện để vẽ biểu đồ

def adaptive_two_pass_median_filter(noisy_image, w1_size=3, w2_size=3, a=1.0, b=1.0):
    """
    Thực hiện Bộ lọc trung vị thích ứng hai lượt.

    Args:
        noisy_image (np.ndarray): Ảnh đầu vào (thang xám) bị nhiễu xung.
                                   Giá trị pixel nên là float hoặc int.
        w1_size (int): Kích thước cửa sổ cho lần lọc trung vị đầu tiên (ví dụ: 3, 5).
        w2_size (int): Kích thước cửa sổ cho lần lọc trung vị thứ hai.
        a (float): Tham số kiểm soát ngưỡng phát hiện cột quá sửa (trong η = a * σλ).
        b (float): Tham số kiểm soát số lượng pixel được khôi phục (trong K).

    Returns:
        np.ndarray: Ảnh đã lọc nhiễu.
    """
    if noisy_image.ndim != 2:
        raise ValueError("Hàm này chỉ hỗ trợ ảnh thang xám (2 chiều).")

    X = noisy_image.astype(float) # Làm việc với float để tránh lỗi tràn số
    M, N = X.shape

    # --- Bước 1: Lọc trung vị lần đầu ---
    Y = median_filter(X, size=w1_size, mode='reflect') # mode='reflect' xử lý biên ảnh

    # Tính ma trận lỗi E1
    diff_xy = X - Y
    E1 = (diff_xy != 0).astype(int)

    # --- Bước 2: Thay thế thích ứng ---
    # Ước tính tỷ lệ lỗi theo cột và các tham số thống kê
    lambda_n = np.mean(E1, axis=0) # Tỷ lệ lỗi trung bình cho mỗi cột (M x N) -> (N,)
    mean_lambda = np.mean(lambda_n) # Trung bình của các tỷ lệ lỗi cột (Λ' trong giải thích)
    sigma_lambda = np.std(lambda_n) # Độ lệch chuẩn của các tỷ lệ lỗi cột

    # Ngưỡng phát hiện
    eta = a * sigma_lambda

    # Khởi tạo ảnh trung gian và ma trận lỗi E2
    Y_tilde = Y.copy()
    E2 = np.zeros_like(X, dtype=int)

    for n in range(N): # Duyệt qua từng cột
        # Kiểm tra nếu cột có khả năng bị sửa đổi quá mức
        if (lambda_n[n] - mean_lambda) > eta:
            # Tính số pixel cần khôi phục K
            K_float = M * (lambda_n[n] - mean_lambda + b * sigma_lambda) # Scale K về số lượng pixel
            K = int(round(K_float))
            K = max(0, min(M, K)) # Đảm bảo K nằm trong khoảng [0, M]

            if K > 0:
                # Tính vector chênh lệch cho cột n
                e_col = X[:, n] - Y[:, n]

                # Tìm chỉ số của K giá trị có độ lớn nhỏ nhất trong e_col
                # Sử dụng argsort để lấy chỉ số sau khi sắp xếp theo giá trị tuyệt đối
                abs_e_col = np.abs(e_col)
                # Chỉ xét những pixel thực sự đã bị thay đổi (E1[m, n] == 1)
                indices_in_col = np.where(E1[:, n] == 1)[0]

                if len(indices_in_col) > 0:
                   # Lấy giá trị tuyệt đối của chênh lệch tại các vị trí đã thay đổi
                   abs_e_at_indices = abs_e_col[indices_in_col]
                   # Sắp xếp các chỉ số này dựa trên độ lớn chênh lệch
                   sorted_indices = indices_in_col[np.argsort(abs_e_at_indices)]
                   # Chọn ra K chỉ số đầu tiên (có chênh lệch nhỏ nhất)
                   v = sorted_indices[:K]

                   # Khôi phục pixel và cập nhật E2
                   if len(v) > 0:
                       Y_tilde[v, n] = X[v, n]
                       E2[v, n] = 1

    # --- Bước 3: Lọc trung vị lần hai ---
    # Áp dụng bộ lọc lên Y_tilde
    Z_temp = median_filter(Y_tilde, size=w2_size, mode='reflect')

    # Tạo ảnh kết quả cuối cùng Z
    # Giữ nguyên giá trị gốc (X hoặc Y_tilde) tại các vị trí E2=1
    # Lấy giá trị từ Z_temp tại các vị trí E2=0
    Z = np.where(E2 == 1, Y_tilde, Z_temp) # Hoặc Z = Z_temp * (1 - E2) + Y_tilde * E2

    # Chuyển về kiểu dữ liệu gốc nếu cần (ví dụ uint8 cho ảnh)
    if np.issubdtype(noisy_image.dtype, np.integer):
        Z = np.clip(Z, 0, 255).astype(noisy_image.dtype) # Giả sử ảnh 8-bit
    elif np.issubdtype(noisy_image.dtype, np.floating):
         Z = Z.astype(noisy_image.dtype)


    return Z

#Thêm nhiễu
def add_salt_and_pepper_noise(image, percentage):
    """
    Thêm nhiễu Salt and Pepper vào ảnh.

    Args:
        image (np.ndarray): Ảnh đầu vào (thang xám).
        percentage (float): Tỷ lệ nhiễu (từ 0.0 đến 1.0).

    Returns:
        np.ndarray: Ảnh đã thêm nhiễu.
    """
    noisy_image = image.copy()
    rows, cols = image.shape
    num_noise_pixels = int(rows * cols * percentage)

    # Chọn ngẫu nhiên các tọa độ để thêm nhiễu
    x_coords = np.random.randint(0, rows, num_noise_pixels)
    y_coords = np.random.randint(0, cols, num_noise_pixels)

    # Chia gần đều số lượng salt (255) và pepper (0)
    num_salt = num_noise_pixels // 2
    num_pepper = num_noise_pixels - num_salt

    # Thêm nhiễu salt (255)
    noisy_image[x_coords[:num_salt], y_coords[:num_salt]] = 255

    # Thêm nhiễu pepper (0)
    noisy_image[x_coords[num_salt:], y_coords[num_salt:]] = 0

    return noisy_image


#Tính MSE và MAE
def calculate_mse(image_original, image_processed):
    """Tính Mean Squared Error (MSE) giữa hai ảnh."""
    if image_original.shape != image_processed.shape:
        raise ValueError("Kích thước hai ảnh phải giống nhau để tính MSE.")
    # Chuyển sang float để tránh lỗi tràn số khi trừ
    err = np.sum((image_original.astype("float") - image_processed.astype("float")) ** 2)
    mse = err / float(image_original.shape[0] * image_original.shape[1])
    return mse

def calculate_mae(image_original, image_processed):
    """Tính Mean Absolute Error (MAE) giữa hai ảnh."""
    if image_original.shape != image_processed.shape:
        raise ValueError("Kích thước hai ảnh phải giống nhau để tính MAE.")
    # Chuyển sang float để tránh lỗi tràn số khi trừ
    err = np.sum(np.abs(image_original.astype("float") - image_processed.astype("float")))
    mae = err / float(image_original.shape[0] * image_original.shape[1])
    return mae

if __name__ == '__main__':
    # --- 1. Tải ảnh gốc (sạch, chưa có nhiễu) ---
    # Thay 'lena_gray.png' bằng đường dẫn đến ảnh thang xám gốc của bạn
    # Bạn có thể tìm ảnh test như 'lena', 'boat', 'baboon' trên mạng
    original_image_path = 'lena.jpg'
    try:
        img_original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
             raise FileNotFoundError(f"Không tìm thấy file ảnh gốc: {original_image_path}")
        print(f"Đã tải ảnh gốc thành công: {original_image_path} - Kích thước: {img_original.shape}")

    except FileNotFoundError as e:
        print(e)
        exit() # Thoát nếu không tải được ảnh gốc
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc ảnh gốc: {e}")
        exit()

    # --- 2. Thiết lập các mức nhiễu và danh sách lưu kết quả ---
    noise_percentages = np.arange(5, 101, 5) # Từ 5% đến 100%, bước nhảy 5%
    mse_results = []
    mae_results = []

    print("\nBắt đầu quá trình thêm nhiễu, lọc và tính lỗi...")
    # --- 3. Vòng lặp qua các mức nhiễu ---
    for percent in noise_percentages:
        noise_fraction = percent / 100.0
        print(f"Đang xử lý nhiễu {percent}%...")

        # Thêm nhiễu vào ảnh gốc
        img_noisy = add_salt_and_pepper_noise(img_original, noise_fraction)

        # Áp dụng bộ lọc thích ứng
        # Bạn có thể thay đổi w1_size, w2_size, a, b nếu muốn thử nghiệm
        img_filtered = adaptive_two_pass_median_filter(img_noisy, w1_size=3, w2_size=3, a=1.0, b=1.0)

        # Tính toán MSE và MAE giữa ảnh gốc và ảnh đã lọc
        current_mse = calculate_mse(img_original, img_filtered)
        current_mae = calculate_mae(img_original, img_filtered)

        # Lưu kết quả
        mse_results.append(current_mse)
        mae_results.append(current_mae)

        # (Tùy chọn) Hiển thị ảnh ở một mức nhiễu nào đó để kiểm tra
        # if percent == 50:
        #     cv2.imshow(f'Anh goc', img_original)
        #     cv2.imshow(f'Nhieu {percent}%', img_noisy)
        #     cv2.imshow(f'Loc voi nhieu {percent}%', img_filtered)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

    print("Hoàn tất quá trình tính toán lỗi.")

    # --- 4. Vẽ biểu đồ ---
    print("Đang vẽ biểu đồ...")
    plt.figure(figsize=(12, 5)) # Kích thước cửa sổ biểu đồ

    # Biểu đồ MSE
    plt.subplot(1, 2, 1) # 1 hàng, 2 cột, vị trí thứ 1
    plt.plot(noise_percentages, mse_results, marker='o', linestyle='-', color='b')
    plt.title('MSE theo Tỷ lệ nhiễu Salt & Pepper')
    plt.xlabel('Tỷ lệ nhiễu Salt & Pepper (%)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(noise_percentages) # Hiển thị rõ các mốc % trên trục x
    plt.grid(True) # Hiện lưới

    # Biểu đồ MAE
    plt.subplot(1, 2, 2) # 1 hàng, 2 cột, vị trí thứ 2
    plt.plot(noise_percentages, mae_results, marker='s', linestyle='--', color='r')
    plt.title('MAE theo Tỷ lệ nhiễu Salt & Pepper')
    plt.xlabel('Tỷ lệ nhiễu Salt & Pepper (%)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xticks(noise_percentages)
    plt.grid(True)

    plt.tight_layout() # Tự động điều chỉnh khoảng cách giữa các biểu đồ
    plt.show() # Hiển thị cửa sổ biểu đồ

    print("Hoàn thành!")