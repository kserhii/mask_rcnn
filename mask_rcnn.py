import os
import time
import random
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np


_COCO_LABELS_FILE = 'object_detection_classes_coco.txt'
_COCO_COLORS_FILE = 'colors.txt'
_COCO_WEIGHTS_FILE = 'frozen_inference_graph.pb'
_COCO_CONFIG_FILE = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

logging.basicConfig(level='INFO')
log = logging.getLogger('rcnn')


def parse_args() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs=1, type=Path,
        help="path to input image")
    ap.add_argument("-m", "--mask-rcnn", type=Path, default=Path('mask-rcnn-coco'),
	    help="base path to mask-rcnn directory")
    ap.add_argument("-v", "--visualize", action='store_true',
	    help="whether or not we are going to visualize each instance")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
	    help="minimum threshold for pixel-wise mask segmentation")
    args = vars(ap.parse_args())

    mask_rcnn_dir = args['mask_rcnn']
    args['image'] = args['image'][0]
    args['labels_path'] = mask_rcnn_dir / _COCO_LABELS_FILE
    args['colors_path'] = mask_rcnn_dir / _COCO_COLORS_FILE
    args['weights_path'] = mask_rcnn_dir / _COCO_WEIGHTS_FILE
    args['config_path'] = mask_rcnn_dir / _COCO_CONFIG_FILE
    
    return args


def main(args):
    log.info('Load COCO dataset from folder %s', args['mask_rcnn'])
    labels = args['labels_path'].read_text().strip().split('\n')
    colors = np.array([
        np.array(c.split(",")).astype("int") 
        for c in args['colors_path'].read_text().strip().split('\n')], dtype='uint8')
    net = cv2.dnn.readNetFromTensorflow(str(args['weights_path']), str(args['config_path']))

    log.info("Load input image: %s", args['image'])
    image = cv2.imread(str(args['image']))
    (imH, imW) = image.shape[:2]

    log.info('Start R-CNN Mask detection for "%s" (%d/%d)', args['image'].name, imW, imH)
    start = time.monotonic()

    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=True)
    net.setInput(blob)
    (boxes, masks) = net.forward(['detection_out_final', 'detection_masks'])

    log.info(f"Done in {time.monotonic() - start:.03f} sec")
    log.info("Found %d mask shapes. Masks shape: %s", boxes.shape[2], masks.shape)

    # loop over the number of detected objects
    for i in range(boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
    
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence <= args["confidence"]:
            log.info("Mask %d (%f) didn't pass the confidence (%f). Continue...", 
                confidence, i, args['confidence'])
            continue

        log.info('Applay mask %d to the image', i)
        # clone our original image so we can draw on it
        img_clone = image.copy()
    
        # scale the bounding box coordinates back relative to the
        # size of the image and then compute the width and the height
        # of the bounding box
        box = boxes[0, 0, i, 3:7] * np.array([imW, imH, imW, imH])
        (startX, startY, endX, endY) = box.astype("int")
        boxW, boxH = endX - startX, endY - startY

        # extract the pixel-wise segmentation for the object, resize
        # the mask such that it's the same dimensions of the bounding
        # box, and then finally threshold to create a *binary* mask
        mask = masks[i, classID]
        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
        mask = (mask > args["threshold"])

        # extract the ROI of the image
        roi = img_clone[startY:endY, startX:endX]

        if args["visualize"]:         
            # convert the mask from a boolean to an integer mask with
            # to values: 0 or 255, then apply the mask
            vis_mask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=vis_mask)

            # show the extracted ROI, the mask, along with the
            # segmented instance
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Mask", vis_mask)
            cv2.imshow("Segmented", instance)

        # now, extract *only* the masked region of the ROI by passing
        # in the boolean mask array as our slice condition
        roi = roi[mask]
        
        # randomly select a color that will be used to visualize this
        # particular instance segmentation then create a transparent
        # overlay by blending the randomly selected color with the ROI
        color = random.choice(colors)
        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

        # store the blended ROI in the original image
        img_clone[startY:endY, startX:endX][mask] = blended

        # draw the bounding box of the instance on the image
        color = [int(c) for c in color]
        cv2.rectangle(img_clone, (startX, startY), (endX, endY), color, 2)

        # draw the predicted label and associated probability of the
        # instance segmentation on the image
        text = "{}: {:.4f}".format(labels[classID], confidence)
        cv2.putText(img_clone, text, (startX, startY - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show the output image
        cv2.imshow("Output (press any key to close)", img_clone)
        cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args()
    main(args)


