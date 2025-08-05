# import os
# import tarfile
# import urllib.request

# # URLs and paths
# VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
# VOC_TAR = "VOCtrainval_11-May-2012.tar"
# EXTRACTED_FOLDER = "VOCdevkit/VOC2012"

# TRAIN_FOLDER = "train"
# VALID_FOLDER = "valid"

# if os.path.exists(TRAIN_FOLDER) and os.path.exists(VALID_FOLDER):
#     print("VOC dataset already downloaded and extracted.")
# else:
#     if not os.path.exists(VOC_TAR):
#         print("Downloading VOC 2012 dataset...")
#         urllib.request.urlretrieve(VOC_URL, VOC_TAR)
#         print("Download complete.")

#     # Extract the tar file
#     print("Extracting dataset...")
#     with tarfile.open(VOC_TAR) as tar:
#         tar.extractall()
#     print("Extraction complete.")

#     
#     os.makedirs(TRAIN_FOLDER, exist_ok=True)
#     os.makedirs(VALID_FOLDER, exist_ok=True)

#     # Copy JPEGImages and Annotations to train and valid folders
#     import shutil
#     from sklearn.model_selection import train_test_split

#     images_dir = os.path.join(EXTRACTED_FOLDER, "JPEGImages")
#     annotations_dir = os.path.join(EXTRACTED_FOLDER, "Annotations")

#     image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
#     train_files, valid_files = train_test_split(image_files, test_size=0.2, random_state=42)

#     for f in train_files:
#         shutil.copy(os.path.join(images_dir, f), os.path.join(TRAIN_FOLDER, f))
#         xml_file = f.replace(".jpg", ".xml")
#         shutil.copy(os.path.join(annotations_dir, xml_file), os.path.join(TRAIN_FOLDER, xml_file))

#     for f in valid_files:
#         shutil.copy(os.path.join(images_dir, f), os.path.join(VALID_FOLDER, f))
#         xml_file = f.replace(".jpg", ".xml")
#         shutil.copy(os.path.join(annotations_dir, xml_file), os.path.join(VALID_FOLDER, xml_file))

#     print("Train and validation data prepared.")


# import torch
# import os
# import urllib.request
# import requests
# import zipfile



# train = 'Train'
# test = 'Test'
# if not os.path.exists(train):
#     os.makedirs(train)
# if not os.path.exists(test):
#     os.makedirs(test)

# def pascal_dataset():
#     # Pascal VOC Dataset
#     url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
#     response = requests.get(url)




import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i


            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

          
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

            
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
    