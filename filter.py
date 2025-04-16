import numpy as np
from scipy.ndimage import median_filter
import cv2 # Hoặc from PIL import Image

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

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # Đọc ảnh (ví dụ: sử dụng OpenCV)
    try:
        img_noisy = cv2.imread('Grayscale-Boats-image-corrupted-with-50-Salt-Pepper-noise.png', cv2.IMREAD_GRAYSCALE) # Thay bằng đường dẫn ảnh của bạn
        if img_noisy is None:
             raise FileNotFoundError("Không tìm thấy file ảnh hoặc định dạng không hỗ trợ.")

        # Áp dụng bộ lọc
        img_filtered_adaptive = adaptive_two_pass_median_filter(img_noisy, w1_size=3, w2_size=3, a=1.0, b=1.0)

        # Áp dụng bộ lọc trung vị chuẩn để so sánh
        img_filtered_median = median_filter(img_noisy, size=3, mode='reflect')

        # Hiển thị hoặc lưu kết quả
        cv2.imshow('Ảnh gốc nhiễu', img_noisy)
        cv2.imshow('Lọc trung vị chuẩn', img_filtered_median.astype(np.uint8))
        cv2.imshow('Lọc trung vị thích ứng hai lượt', img_filtered_adaptive.astype(np.uint8))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite('filtered_adaptive.png', img_filtered_adaptive)
        # cv2.imwrite('filtered_median.png', img_filtered_median)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

    # Ví dụ với dữ liệu numpy tự tạo
    print("\nVí dụ với dữ liệu tự tạo:")
    np.random.seed(0)
    test_data = np.zeros((10, 10)) + 100
    # Thêm nhiễu xung
    noise_level = 0.2
    coords = np.random.randint(0, 10, size=(int(10*10*noise_level), 2))
    vals = np.random.choice([0, 255], size=len(coords))
    test_data[coords[:, 0], coords[:, 1]] = vals
    print("Dữ liệu nhiễu:\n", test_data.astype(int))

    filtered_data = adaptive_two_pass_median_filter(test_data, w1_size=3, w2_size=3, a=1.0, b=1.0)
    print("\nDữ liệu sau khi lọc:\n", filtered_data.astype(int))