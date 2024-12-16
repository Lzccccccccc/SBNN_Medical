# import torch
# import torch.nn as nn
#
#
# class FeatureExtractionModule(nn.Module):
#     def __init__(self, padding_3x3=1, padding_5x5=2):
#         super(FeatureExtractionModule, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), padding=padding_3x3)
#         self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), padding=padding_5x5)
#         self.conv3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), padding=padding_3x3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x
#
#
# fe = FeatureExtractionModule()
# input_data = torch.randn(1, 3, 224, 224)
# print(fe(input_data).size())
import torch
import torch.nn.functional as F

# Assuming your input tensor is named 'image_tensor'
# Shape: 1 * 224 * 56 * 56
image_tensor = torch.randn(1, 224, 56, 56)

# Define the target size
target_size = (224, 224)
scale_factors = (224 / image_tensor.size(2), 224 / image_tensor.size(3))
resized_image = F.interpolate(image_tensor, scale_factor=scale_factors, mode='bilinear', align_corners=False)

print(resized_image.shape)  # Output: torch.Size([1, 224, 224, 224])
