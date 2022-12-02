from .report import Report
from dapi.utils import image_to_tensor
import cv2
import numpy as np
import torch.nn.functional as F


def run_inference(classifier, im):
    """
    Net: network object
    input_image: Normalized 2D input image.
    """
    im_tensor = image_to_tensor(im)
    class_probs = F.softmax(classifier(im_tensor), dim=1)
    return class_probs


def create_mask(
        attribution,
        threshold,
        sigma=11,
        struc=10,
        channel_wise=False):

    channels, _, _ = attribution.shape

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (struc, struc))

    mask_size = 0
    mask = []

    # construct mask channel by channel
    for c in range(channels):

        # threshold
        if channel_wise:
            channel_mask = (attribution[c, :, :] > threshold)
        else:
            channel_mask = np.any(attribution > threshold, axis=0)

        # morphological closing
        channel_mask = cv2.morphologyEx(
            channel_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel)

        mask_size += np.sum(channel_mask)

        # blur
        mask.append(
            cv2.GaussianBlur(
                channel_mask.astype(np.float32),
                (sigma, sigma),
                0))

    return np.array(mask), mask_size


def evaluate(
        attribution,
        real,
        fake,
        real_class,
        fake_class,
        classifier,
        sigma=11,
        struc=10,
        channel_wise=False,
        num_thresholds=200):

    # copy parts of "real" into "fake", see how much the classification of
    # "fake" changes into "real_class"

    num_thresholds = 200

    classification_fake = run_inference(classifier, fake)[0]

    report = Report(
        attribution,
        real,
        fake)

    for threshold in np.arange(-1.0, 1.0, 2.0/num_thresholds):

        # soft mask of the parts to copy
        mask, mask_size = create_mask(
            attribution,
            threshold,
            sigma,
            struc,
            channel_wise)

        real_masked = real * mask
        fake_masked = fake * mask
        diff = real_masked - fake_masked

        # hybrid = real parts copied to fake
        hybrid = real * mask + fake * (1.0 - mask)

        classification_hybrid = run_inference(classifier, hybrid)[0]

        score_change = (
            classification_hybrid[real_class] -
            classification_fake[real_class])

        report.add_threshold(
            threshold,
            mask,
            mask_size,
            score_change,
            hybrid,
            real_masked,
            fake_masked,
            diff)

    return report
