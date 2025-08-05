import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import My_Model
from dataset import VOCDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)


from model import LossFunction

seed = 42
torch.manual_seed(seed)

lr = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
weight_decay = 0
epochs = 200
num_workers = 2
pin_memory = True
load_model = False
load_model_file = "my_checkpoint.pth.tar"

img_dir = "dataset_voc/images"
label_dir = "dataset_voc/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for i in self.transforms:
            img, bboxes = i(img), bboxes
        return img, bboxes
            
transforms = Compose([transforms.Resize((448, 448)),transforms.ToTensor()])

def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    model.train()
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = loss_fn(output, y)

        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())

    print(f"Mean Loss: {sum(mean_loss) / len(mean_loss)}")

def main():
    model = My_Model(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = LossFunction()

    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)
        
        
    train_dataset = VOCDataset(
        "dataset_voc/train.csv",
        img_dir=img_dir,
        label_dir=label_dir,
        transform=transforms,
    )

    test_dataset = VOCDataset(
        "dataset_voc/test.csv",
        img_dir=img_dir,
        label_dir=label_dir,
        transform=transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


    for epoch in range(epochs):
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        train(train_loader, model, optimizer, loss_fn)
        if epoch % 10 == 0:
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            })
            
        map_score = mean_average_precision(
            pred_boxes,
            target_boxes,
            iou_threshold=0.5,
            box_format="midpoint",
        )

        print(f"Epoch {epoch + 1}/{epochs}, mAP: {map_score.item() * 100:.2f}%")
        

if __name__ == "__main__":
    main()