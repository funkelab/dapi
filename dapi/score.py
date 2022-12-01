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


def get_dapi_score(
        attribution,
        real_img,
        fake_img,
        real_class,
        fake_class,
        classifier,
        sigma=11,
        struc=10,
        channel_wise=False,
        num_thresholds=200):

    # copy parts of "real" into "fake", see how much the classification of
    # "fake" changes into "real_class"

    result_dict = {}
    img_names = [
        "attr",
        "real",
        "fake",
        "hybrid",
        "mask_real",
        "mask_fake",
        "mask_residual",
        "mask_weight"]
    imgs_all = []

    num_thresholds = 200

    classification_fake = run_inference(classifier, fake_img)[0]

    for threshold in np.arange(-1.0, 1.0, 2.0/num_thresholds):

        # soft mask of the parts to copy
        mask, mask_size = create_mask(
            attribution,
            threshold,
            sigma,
            struc,
            channel_wise)

        real_masked = real_img * mask
        fake_masked = fake_img * mask
        diff_img = real_masked - fake_masked

        # hybrid = real parts copied to fake
        hybrid_img = real_img * mask + fake_img * (1.0 - mask)

        classification_hybrid = run_inference(classifier, hybrid_img)[0]

        imgs = [
            attribution,
            real_img,
            fake_img,
            hybrid_img,
            real_masked,
            fake_masked,
            diff_img,
            mask
        ]

        imgs_all.append(imgs)

        score_change = (
            classification_hybrid[real_class] -
            classification_fake[real_class])
        result_dict[threshold] = [
            float(score_change.detach().cpu().numpy()),
            mask_size
        ]

    return result_dict, img_names, imgs_all
