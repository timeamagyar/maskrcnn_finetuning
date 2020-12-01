import os
import torch
import torch.utils.data
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import numpy as np
import functools
import operator


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target annotations file for an image
        annotations = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        img_width, img_height = img.size

        # number of objects in the image
        num_objs = len(annotations)
        instance_masks = []
        print("image_id", img_id)
        for annotation in annotations:
            # this is our canvas
            mask = Image.new('1', (img_width, img_height))
            mask_draw = ImageDraw.Draw(mask, '1')

            segmentation = functools.reduce(operator.iconcat, annotation['segmentation'], [])
            mask_draw.polygon(segmentation, fill=1)
            bool_array = np.array(mask) > 0
            instance_masks.append(bool_array)

        masks = torch.as_tensor(instance_masks, dtype=torch.uint8)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = annotations[i]['bbox'][0]
            ymin = annotations[i]['bbox'][1]
            xmax = xmin + annotations[i]['bbox'][2]
            ymax = ymin + annotations[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(annotations[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
