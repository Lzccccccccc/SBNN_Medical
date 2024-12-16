from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, resnet18
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
from models.utils import preprocess_enhance


def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(20, 20))
    if titles == False:
        titles = "0123456789"
    # cv2.imwrite('grad_cam.jpg', imgs[len(imgs) - 1])
    plt.imshow(imgs[len(imgs) - 1])
    # for i in range(1, lens + 1):
    #     cols = 100 + lens * 10 + i
    #     plt.xticks(())
    #     plt.yticks(())
    #     plt.subplot(cols)
    #     if len(imgs[i - 1].shape) == 2:
    #         plt.imshow(imgs[i - 1], cmap='Reds')
    #     else:
    #         plt.savefig(imgs[i - 1])
    #         plt.imshow(imgs[i - 1])
    #     plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr


path = "8777.png"
bin_data = Image.open(path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Lambda(preprocess_enhance),
    transforms.ToTensor()
])

input_tensors = preprocess(bin_data).unsqueeze(0)
model = resnet18(pretrained=False)
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('best_model2.pth', map_location=torch.device('cpu')))
target_layers = [model.layer4[-1]]

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensors)  # targets=None 自动调用概率最大的类别显示
    for grayscale_cam, tensor in zip(grayscale_cams, input_tensors):
        rgb_img = tensor2img(tensor)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        myimshows([rgb_img, grayscale_cam, visualization], ["image", "cam", "image + cam"])
