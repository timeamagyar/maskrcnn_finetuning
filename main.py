import torch.utils.data
from PIL import Image
from torchvision import transforms
import cv2
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from coco import CocoDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import os

def check_mem():
    mem = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split(",")
    return mem

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# In my case, just added ToTensor
def get_transform():
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)



def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_prediction(img_path, threshold):
    global model, device
    img = Image.open(img_path) # Load the image
    transform = transforms.Compose([
    transforms.ToTensor()])
    img = transform(img) # Apply the transform to the image
    pred = model([img.to(device)]) # Pass the image to the model
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    print(pred_score)
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    # bugfix: only squeeze along axis 1, if model is strong we might get 4 dimensional tensor of the form (1, 1, width, height)
    masks = (pred[0]['masks']>0.5).squeeze(axis=1).detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

    masks = masks[:pred_t + 1]
    masks = np.swapaxes(masks, 0, 2)
    masks = np.swapaxes(masks, 0, 1)
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def instance_segmentation_api(img_path, threshold=0.3, rect_th=3, text_size=3, text_th=3):
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks[0][0])):
        rgb_mask = random_colour_masks(masks[:, :, i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("result.png")


if __name__=='__main__':
    global model, device
    # check model performance by feeding sample images to Mask-RCNN
if 0:
    # read example image to test segmentation performance
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-image_path",
        type=str, help="path to color image to be fed to Mask-RCNN"
    )

    args = parser.parse_args()
    image_path = args.image_path

    # Now let's instantiate the model and the optimizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)
    model.load_state_dict(torch.load('./maskrcnn.pt'))
    model.eval()

    # segment example image, only accept matches above a confidence threshold of 0.9
    instance_segmentation_api(image_path, threshold=0.9)

    # enter training mode
elif 1:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-train_data_path",
        type=str, help="path to raw color image data used for training"
    )

    parser.add_argument(
        "-train_coco_path",
        type=str, help="path to coco annotations"
    )

    args = parser.parse_args()
    train_data_dir = args.train_data_path
    train_coco = args.train_coco_path

    # create own Dataset
    full_dataset = CocoDataset(root=train_data_dir,
                               annotation=train_coco,
                               transforms=get_transform())

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    dataset, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    torch.cuda.empty_cache()
    import torch

    # Now let's instantiate the model and the optimizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)
    #model.load_state_dict(torch.load('./maskrcnn_backup.pt'), strict=False)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset after every epoch
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), './maskrcnn.pt')
