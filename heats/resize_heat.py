from PIL import Image


def resize_image(input_path, output_path, size=(600, 600)):
    with Image.open(input_path) as img:
        img = img.resize(size)
        img.save(output_path)


# 使用示例
resize_image('pneumonia_heatmap.jpg', 'pneumonia_heatmap.jpg')
