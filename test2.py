import numpy as np
import cv2

# 读取图像
image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)  # 读取灰度图像

# 定义拉伸的最小和最大像素值
min_pixel_value = 0
max_pixel_value = 255

# 计算图像的最小和最大像素值
min_intensity = np.min(image)
max_intensity = np.max(image)

# 对比度拉伸
stretched_image = ((image - min_intensity) / (max_intensity - min_intensity)) * (max_pixel_value - min_pixel_value) + min_pixel_value

# 转换数据类型为8位无符号整数
stretched_image = np.uint8(stretched_image)

# 保存拉伸后的图像
cv2.imwrite('stretched_image.png', stretched_image)

# 显示原始图像和拉伸后的图像
