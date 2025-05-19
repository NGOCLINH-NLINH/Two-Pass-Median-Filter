import numpy as np
from scipy.ndimage import median_filter
import cv2 # Hoặc from PIL import Image
import matplotlib.pyplot as plt # Thư viện để vẽ biểu đồ
from atpmf import atpmf_grayscale
from utils import calculate_mae
from utils import calculate_mse
from utils import add_salt_and_pepper_noise
from utils import convert_to_grayscale

if __name__ == '__main__':
    # --- 1. Tải ảnh gốc (sạch, chưa có nhiễu) ---
    # Thay 'lena_gray.png' bằng đường dẫn đến ảnh thang xám gốc của bạn
    # Bạn có thể tìm ảnh test như 'lena', 'boat', 'baboon' trên mạng
    original_image_path = 'images/lena_color.png'
    try:
        img_original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
             raise FileNotFoundError(f"Không tìm thấy file ảnh gốc: {original_image_path}")
        print(f"Đã tải ảnh gốc thành công: {original_image_path} - Kích thước: {img_original.shape}")

        # Chuyển về grayscale
        img_original = convert_to_grayscale(img_original)

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
        img_filtered = atpmf_grayscale(img_noisy, W1_size=3, W2_size=3, a=1.0, b=1.0)

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