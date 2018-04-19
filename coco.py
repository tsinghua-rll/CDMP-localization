import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np

COCO_ROOT = osp.join('data/coco/')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
# COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                 'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
#                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                 'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
#                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                 'teddy bear', 'hair drier', 'toothbrush')

COCO_CLASSES = {1 : 'person', 2 : 'bicycle', 3 : 'car', 4 : 'motorcycle', 5 : 'airplane', 6 : 'bus', 7 : 'train', 8 : 'truck', 9 : 'boat', 10 : 'traffic light', 11 : 'fire hydrant', 13 : 'stop sign', 14 : 'parking meter', 15 : 'bench', 16 : 'bird', 17 : 'cat', 18 : 'dog', 19 : 'horse', 20 : 'sheep', 21 : 'cow', 22 : 'elephant', 23 : 'bear', 24 : 'zebra', 25 : 'giraffe', 27 : 'backpack', 28 : 'umbrella', 31 : 'handbag', 32 : 'tie', 33 : 'suitcase', 34 : 'frisbee', 35 : 'skis', 36 : 'snowboard', 37 : 'sports ball', 38 : 'kite', 39 : 'baseball bat', 40 : 'baseball glove', 41 : 'skateboard', 42 : 'surfboard', 43 : 'tennis racket', 44 : 'bottle', 46 : 'wine glass', 47 : 'cup', 48 : 'fork', 49 : 'knife', 50 : 'spoon', 51 : 'bowl', 52 : 'banana', 53 : 'apple', 54 : 'sandwich', 55 : 'orange', 56 : 'broccoli', 57 : 'carrot', 58 : 'hot dog', 59 : 'pizza', 60 : 'donut', 61 : 'cake', 62 : 'chair', 63 : 'couch', 64 : 'potted plant', 65 : 'bed', 67 : 'dining table', 70 : 'toilet', 72 : 'tv', 73 : 'laptop', 74 : 'mouse', 75 : 'remote', 76 : 'keyboard', 77 : 'cell phone', 78 : 'microwave', 79 : 'oven', 80 : 'toaster', 81 : 'sink', 82 : 'refrigerator', 84 : 'book', 85 : 'clock', 86 : 'vase', 87 : 'scissors', 88 : 'teddy bear', 89 : 'hair drier', 90 : 'toothbrush'}

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        # self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))
        pass

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                # label_idx = self.label_map[obj['category_id']] - 1
                label_idx = obj['category_id']
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='trainval35k', transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        sys.path.append(osp.join(root, COCO_API))
        from pycocotools.coco import COCO
        self.root = osp.join(root, IMAGES, image_set)
        self.coco = COCO(osp.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(osp.join(self.root, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class COCOLocalizationAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        pass

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = obj['category_id']
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCOLocalization(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='train2017', transform=None,
                 target_transform=COCOLocalizationAnnotationTransform(), dataset_name='MS COCO', object_size=224):
        # sys.path.append(osp.join(root, COCO_API))
        from pycocotools.coco import COCO
        self.root = osp.join(root, image_set)
        self.coco = COCO(osp.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.object_size=object_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        h, w = im.shape[:2]
        # im (H, W, C) -> ((C, H, W), (C, H, W)) (random select target)
        # gt (num, 5) -> (2) 
        num = len(gt)
        gt = gt[np.random.randint(0, num)]
        xmin, ymin, xmax, ymax, label_id = gt
        xmin = int(xmin*(h-1))
        xmax = int(xmax*(h-1)) 
        ymin = int(ymin*(w-1))
        ymax = int(ymax*(w-1))
        if xmin == xmax:
            if xmin == 0:
                xmax += 1
            elif xmax == h-1:
                xmin -= 1
        if ymin == ymax:
            if ymin == 0:
                ymax += 1
            elif ymax == w-1:
                ymin -= 1

        object_img = torch.from_numpy(cv2.resize(im[ymin:ymax, xmin:xmax, :], (self.object_size, 
            self.object_size))).permute(2,0,1)
        im = torch.from_numpy(im).permute(2,0,1)
        gt = [(xmin + xmax)/2/h, (ymin + ymax)/2/w, label_id]
        return im, object_img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(osp.join(self.root, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
