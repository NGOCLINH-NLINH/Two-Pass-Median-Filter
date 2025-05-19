import numpy as np
import cv2

def add_salt_and_pepper_noise(image, noise_percentage):
    """
    Thêm nhiễu Salt & Pepper vào ảnh (tuân theo phân phối nhị thức).

    Tham số:
        image (numpy.ndarray): Ảnh đầu vào (grayscale hoặc màu, dtype uint8).
        noise_percentage (float): Xác suất mỗi pixel bị nhiễu (0.0 đến 1.0).

    Trả về:
        numpy.ndarray: Ảnh bị nhiễu.
    """
    noisy_image = np.copy(image)
    original_dtype = image.dtype

    if image.ndim == 2:  # Grayscale
        height, width = image.shape
        # Tạo mask nhiễu với xác suất p cho mỗi pixel
        rand = np.random.rand(height, width)
        salt_mask = rand < (noise_percentage / 2)
        pepper_mask = (rand >= (noise_percentage / 2)) & (rand < noise_percentage)

        noisy_image[salt_mask] = 255
        noisy_image[pepper_mask] = 0

    elif image.ndim == 3 and image.shape[2] == 3:  # Màu RGB
        height, width, _ = image.shape
        rand = np.random.rand(height, width)
        salt_mask = rand < (noise_percentage / 2)
        pepper_mask = (rand >= (noise_percentage / 2)) & (rand < noise_percentage)

        noisy_image[salt_mask] = [255, 255, 255]
        noisy_image[pepper_mask] = [0, 0, 0]

    else:
        raise ValueError("Chỉ hỗ trợ ảnh grayscale (2D) hoặc ảnh màu RGB (3D với 3 kênh).")

    return noisy_image.astype(original_dtype)

def add_gaussian_noise(image, std_dev, noise_percentage):
    """
    Thêm nhiễu Gaussian lên một tỷ lệ phần trăm pixel trong ảnh (giống Salt & Pepper style).

    Args:
        image (np.ndarray): Ảnh đầu vào (grayscale hoặc RGB).
        std_dev (float): Độ lệch chuẩn của nhiễu Gaussian.
        noise_percentage (float): Tỷ lệ pixel bị nhiễu (0.0 đến 1.0).

    Returns:
        np.ndarray: Ảnh bị nhiễu Gaussian một phần.
    """
    noisy_image = image.astype(np.float64)
    height, width = image.shape[:2]
    total_pixels = height * width
    num_noisy = int(noise_percentage * total_pixels)

    flat_indices = np.random.choice(total_pixels, num_noisy, replace=False)
    rows = flat_indices // width
    cols = flat_indices % width

    if image.ndim == 2:  # Grayscale
        noise = np.random.normal(0, std_dev, num_noisy)
        noisy_image[rows, cols] += noise

    elif image.ndim == 3 and image.shape[2] == 3:  # Color
        noise = np.random.normal(0, std_dev, (num_noisy, 3))
        noisy_image[rows, cols, :] += noise

    else:
        raise ValueError("Ảnh đầu vào phải là grayscale (2D) hoặc RGB (3D với 3 kênh).")

    # Clip và trả về kiểu ban đầu
    if np.issubdtype(image.dtype, np.integer):
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(image.dtype)
    else:
        return np.clip(noisy_image, 0.0, 255.0)

def add_speckle_noise(image, variance, noise_percentage):
    """
    Thêm nhiễu Speckle (nhân) lên một tỷ lệ phần trăm pixel (giống Salt & Pepper style).

    Args:
        image (np.ndarray): Ảnh đầu vào (grayscale hoặc RGB).
        variance (float): Phương sai của nhiễu Gaussian nhân.
        noise_percentage (float): Tỷ lệ pixel bị nhiễu (0.0 đến 1.0).

    Returns:
        np.ndarray: Ảnh bị nhiễu Speckle một phần.
    """
    noisy_image = image.astype(np.float64)
    height, width = image.shape[:2]
    total_pixels = height * width
    num_noisy = int(noise_percentage * total_pixels)

    flat_indices = np.random.choice(total_pixels, num_noisy, replace=False)
    rows = flat_indices // width
    cols = flat_indices % width

    std_dev = np.sqrt(variance)

    if image.ndim == 2:  # Grayscale
        noise = np.random.normal(0, std_dev, num_noisy)
        noisy_image[rows, cols] += noisy_image[rows, cols] * noise

    elif image.ndim == 3 and image.shape[2] == 3:  # Color
        noise = np.random.normal(0, std_dev, (num_noisy, 3))
        noisy_image[rows, cols, :] += noisy_image[rows, cols, :] * noise

    else:
        raise ValueError("Ảnh đầu vào phải là grayscale (2D) hoặc RGB (3D với 3 kênh).")

    # Clip và trả về kiểu ban đầu
    if np.issubdtype(image.dtype, np.integer):
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(image.dtype)
    else:
        return np.clip(noisy_image, 0.0, 255.0)



# --- Các hàm tính lỗi ---
def calculate_mae(image_original, image_processed):
    """
    Tính Mean Absolute Error (MAE) giữa hai ảnh.
    Ưu tiên tính trên ảnh màu nếu cả hai ảnh là màu và cùng kích thước.
    Nếu không, hoặc nếu kích thước khác nhau, sẽ cố gắng chuyển về thang xám để so sánh.
    """
    # Trường hợp 1: Cả hai ảnh là ảnh màu và có cùng kích thước
    if image_original.ndim == 3 and image_processed.ndim == 3 and image_original.shape == image_processed.shape:
        print("Cả hai ảnh là ảnh màu và cùng kích thước. Tính MAE trên ảnh màu.")
        err = np.sum(np.abs(image_original.astype("float") - image_processed.astype("float")))
        mae = err / float(image_original.size) # image_original.size bao gồm tất cả các pixel trên tất cả các kênh
        return mae

    # Trường hợp 2: Một trong hai ảnh là ảnh xám hoặc kích thước khác nhau, cần chuẩn hóa về xám
    img_orig_final = image_original
    img_proc_final = image_processed

    # Chuyển ảnh gốc sang xám nếu nó là màu và ảnh xử lý là xám (hoặc ngược lại nếu cần)
    if image_original.ndim == 3 and image_processed.ndim == 2:
        print("Ảnh gốc màu, ảnh xử lý xám. Chuyển ảnh gốc sang xám.")
        img_orig_final = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    elif image_original.ndim == 2 and image_processed.ndim == 3:
        print("Ảnh gốc xám, ảnh xử lý màu. Chuyển ảnh xử lý sang xám.")
        img_proc_final = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
    # Nếu cả hai đều màu nhưng kích thước khác nhau (ít khả năng cho MAE/MSE trực tiếp)
    # hoặc một trong hai không phải là ảnh (ví dụ ndim < 2)
    # thì nên có xử lý lỗi cụ thể hơn hoặc quyết định cách chuẩn hóa.
    # Hiện tại, nếu kích thước khác nhau và không rơi vào trường hợp trên, sẽ cố gắng chuyển cả hai sang xám.

    # Nếu sau các bước trên, một trong hai vẫn là màu, chuyển nốt sang xám
    if img_orig_final.ndim == 3:
        print("Chuyển ảnh gốc (có thể đã qua xử lý) sang xám.")
        img_orig_final = cv2.cvtColor(img_orig_final, cv2.COLOR_BGR2GRAY)
    if img_proc_final.ndim == 3:
        print("Chuyển ảnh xử lý (có thể đã qua xử lý) sang xám.")
        img_proc_final = cv2.cvtColor(img_proc_final, cv2.COLOR_BGR2GRAY)

    # Kiểm tra lại kích thước sau khi đã cố gắng chuyển đổi
    if img_orig_final.shape != img_proc_final.shape:
        # Nếu kích thước vẫn không khớp, có thể raise lỗi hoặc quyết định một chiến lược khác
        # Ví dụ: resize một trong hai ảnh (nhưng điều này có thể ảnh hưởng đến độ chính xác của lỗi)
        # Ở đây, dựa theo logic code gốc, ta sẽ ưu tiên đảm bảo chúng cùng số chiều để tính.
        # Nếu một cái là màu một cái xám mà kích thước điểm ảnh bằng nhau thì code gốc sẽ lỗi ở đoạn cvtColor
        # Giả định là nếu đến đây mà shape khác nhau thì không so sánh được.
        raise ValueError(f"Kích thước ảnh không khớp sau khi chuyển đổi và không thể tính toán: {img_orig_final.shape} vs {img_proc_final.shape}")

    print("Tính MAE trên ảnh thang xám.")
    err = np.sum(np.abs(img_orig_final.astype("float") - img_proc_final.astype("float")))
    # Sử dụng size của ảnh đã được chuẩn hóa (có thể là xám)
    mae = err / float(img_orig_final.size)
    return mae

def calculate_mse(image_original, image_processed):
    """
    Tính Mean Squared Error (MSE) giữa hai ảnh.
    Ưu tiên tính trên ảnh màu nếu cả hai ảnh là màu và cùng kích thước.
    Nếu không, hoặc nếu kích thước khác nhau, sẽ cố gắng chuyển về thang xám để so sánh.
    """
    # Trường hợp 1: Cả hai ảnh là ảnh màu và có cùng kích thước
    if image_original.ndim == 3 and image_processed.ndim == 3 and image_original.shape == image_processed.shape:
        print("Cả hai ảnh là ảnh màu và cùng kích thước. Tính MSE trên ảnh màu.")
        # Tính toán trên ảnh màu
        # Đảm bảo kiểu dữ liệu là float để tránh tràn số khi bình phương và cộng dồn
        err = np.sum((image_original.astype("float") - image_processed.astype("float")) ** 2)
        # image_original.size bao gồm tất cả các pixel trên tất cả các kênh (cao * rộng * số kênh)
        mse = err / float(image_original.size)
        return mse

    # Trường hợp 2: Một trong hai ảnh là ảnh xám hoặc kích thước khác nhau, cần chuẩn hóa về xám
    img_orig_final = image_original
    img_proc_final = image_processed

    # Chuyển ảnh gốc sang xám nếu nó là màu và ảnh xử lý là xám (hoặc ngược lại nếu cần)
    if image_original.ndim == 3 and image_processed.ndim == 2:
        print("Ảnh gốc màu, ảnh xử lý xám. Chuyển ảnh gốc sang xám.")
        img_orig_final = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    elif image_original.ndim == 2 and image_processed.ndim == 3:
        print("Ảnh gốc xám, ảnh xử lý màu. Chuyển ảnh xử lý sang xám.")
        img_proc_final = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
    # Các trường hợp khác cần chuẩn hóa hoặc gây lỗi sẽ được xử lý bên dưới

    # Nếu sau các bước trên, một trong hai vẫn là màu (và cái kia có thể đã được chuyển thành xám
    # hoặc cả hai đều màu nhưng kích thước ban đầu khác nhau), chuyển nốt sang xám.
    # Mục tiêu là đưa cả hai về thang xám để so sánh nếu không thể so sánh màu.
    if img_orig_final.ndim == 3:
        print("Chuyển ảnh gốc (có thể đã qua xử lý) sang xám.")
        img_orig_final = cv2.cvtColor(img_orig_final, cv2.COLOR_BGR2GRAY)
    if img_proc_final.ndim == 3:
        print("Chuyển ảnh xử lý (có thể đã qua xử lý) sang xám.")
        img_proc_final = cv2.cvtColor(img_proc_final, cv2.COLOR_BGR2GRAY)

    # Kiểm tra lại kích thước sau khi đã cố gắng chuyển đổi
    if img_orig_final.shape != img_proc_final.shape:
        # Nếu kích thước vẫn không khớp, không thể tính toán.
        raise ValueError(f"Kích thước ảnh không khớp sau khi chuyển đổi và không thể tính toán MSE: {img_orig_final.shape} vs {img_proc_final.shape}")

    print("Tính MSE trên ảnh thang xám.")
    # Tính toán trên ảnh thang xám
    err = np.sum((img_orig_final.astype("float") - img_proc_final.astype("float")) ** 2)
    # Sử dụng size của ảnh đã được chuẩn hóa (chắc chắn là xám ở bước này nếu vào nhánh này)
    mse = err / float(img_orig_final.size)
    return mse

def convert_to_grayscale(image):
    """
    Chuyển ảnh bất kỳ (màu hoặc xám) về ảnh grayscale.
    Nếu ảnh đã là grayscale thì giữ nguyên.
    """
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()