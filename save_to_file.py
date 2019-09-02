import os
import pickle as pkl
import cv2
import numpy as np


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = []
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes.append(np.array([y1, x1, y2, x2]))
    return boxes#boxes.astype(np.int32)


files_pkl = os.listdir("./data/pkl_files")

for indx, file_ in enumerate(files_pkl):
    masks = None
    fullmammo_img = None
    if("pkl" not in file_):
        continue
    with open(os.path.join("./data/pkl_files", file_), 'rb') as f:
        temp = pkl.load(f)
        fullmammo_img = temp['fullmammo_img']
        masks = np.array(temp['masks'])
        masks = np.transpose(masks, [1, 2, 0])

    b_boxes = extract_bboxes(masks)

    for ii in range(len(b_boxes)):
        yy1, xx1, yy2, xx2 = b_boxes[ii]
        image_ = fullmammo_img[int(0.85*yy1):min(int(1.15*yy2), fullmammo_img.shape[0]), int(0.85*xx1):min(int(1.15*xx2), fullmammo_img.shape[1])]
        mask_ = masks[int(0.85*yy1):min(int(1.15*yy2), fullmammo_img.shape[0]), int(0.85*xx1):min(int(1.15*xx2), fullmammo_img.shape[1]), ii]
        if "Training" in file_:
          cv2.imwrite("./data/breast/train/image/"+file_+".jpg", image_)
          cv2.imwrite("./data/breast/train/label/"+file_+".jpg", mask_)
        elif "Test" in file_:
          cv2.imwrite("./data/breast/test/image/"+file_+".jpg", image_)
          cv2.imwrite("./data/breast/test/label/"+file_+".jpg", mask_)