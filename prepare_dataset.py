import os
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split


def get_data_loader():
    labels_confi = ['Atelectasis', 'Mass', 'Infiltration', 'Edema', 'Pneumothorax', 'Nodule', 'Cardiomegaly',
                    'Fibrosis', 'Pneumonia', 'No Finding', 'Consolidation', 'Emphysema', 'Hernia', 'Pleural_Thickening',
                    'Effusion']
    label_to_index = {label: idx for idx, label in enumerate(labels_confi)}
    data = []
    labels = []
    # i = 0
    df_labels = pd.read_csv("data/sample_labels.csv")
    for image in os.listdir("data/images"):
        # if i == 10:
        #     break
        real_path = os.path.join("data/images", image)
        img = Image.open(real_path).convert("RGB")
        data.append(img)
        multiple_labels = df_labels[df_labels["Image Index"] == image]["Finding Labels"].values[0].split("|")
        one_hot = torch.zeros(len(labels_confi))
        for label in multiple_labels:
            one_hot[label_to_index[label]] = 1
        labels.append(one_hot)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test
