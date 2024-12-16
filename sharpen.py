import cv2
import numpy as np

# 读取经过对比度增强的图像
enhanced_image = cv2.imread('enhanced_image.png')


# 颜色校正
def color_correction(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    l_equ = cv2.equalizeHist(l)
    image_lab_equ = cv2.merge((l_equ, a, b))
    corrected_image = cv2.cvtColor(image_lab_equ, cv2.COLOR_LAB2BGR)
    return corrected_image


# 图像锐化
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


color_corrected_image = color_correction(enhanced_image)

sharpened_image = sharpen_image(color_corrected_image)

cv2.imwrite('post_processed_image.jpg', sharpened_image)

cv2.imshow('Post-processed Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
