import cv2
import numpy as np

# 读取输入图片
input_image = cv2.imread("output_file.jpg")

# 检查图片是否成功读取
if input_image is not None:
    # 创建一个与输入图片大小相同的蓝色底图
    blue_background = np.zeros_like(input_image)
    blue_background[:, :] = [255, 0, 0]  # 设置为蓝色

    # 将输入图片与蓝色底图相加
    output_image = cv2.addWeighted(input_image, 0.7, blue_background, 0.3, 0)
    cv2.imwrite("output_heat.jpg", output_image)
    # 显示结果
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to read the input image.")
