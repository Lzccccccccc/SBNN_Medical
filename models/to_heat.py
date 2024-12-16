import cv2
from matplotlib import pyplot as plt

# 读取图片
image_path = 'Pneumonia.jpeg'  # 替换为你的图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图片

# 确保图片是在 0 到 255 范围内

# 应用颜色映射
colored_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

# 显示结果
plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))  # 转换颜色空间以正确显示
plt.axis('off')
plt.show()

# 保存结果图片
# cv2.imwrite('colored_image.jpg', colored_image
