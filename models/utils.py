import cv2
import numpy as np
from PIL import Image


def color_correction(image):
    l_equ = cv2.equalizeHist(image)
    return l_equ


# 图像锐化
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def preprocess_enhance(image):
    input_image = np.array(image)
    # Preprocessing (e.g., noise removal and edge detection)
    # You can replace this with your specific preprocessing steps
    preprocessed_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # preprocessed_image = cv2.GaussianBlur(preprocessed_image, (5, 5), 0)
    # Adaptive Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(preprocessed_image)
    # Contrast Stretching
    # 定义拉伸的最小和最大像素值
    min_pixel_value = 0
    max_pixel_value = 255
    # 计算图像的最小和最大像素值
    min_intensity = np.min(enhanced_image)
    max_intensity = np.max(enhanced_image)
    # 对比度拉伸
    stretched_image = ((enhanced_image - min_intensity) / (max_intensity - min_intensity)) * (
                max_pixel_value - min_pixel_value) + min_pixel_value
    # 转换数据类型为8位无符号整数
    stretched_image = np.uint8(stretched_image)
    # cv2.imwrite('t.jpg', stretched_image)
    # stretched_image = cv2.imread("t.jpg")
    color_corrected_image = color_correction(stretched_image)

    # 执行图像锐化
    sharpened_image = sharpen_image(color_corrected_image)
    return cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2RGB)
    # return Image.fromarray(cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2RGB))
    # return sharpened_image

image = cv2.imread('input.png')
print(preprocess_enhance(image))