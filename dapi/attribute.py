from captum.attr import IntegratedGradients, Saliency, DeepLift,\
                        GuidedGradCam, InputXGradient,\
                        LayerGradCam, GuidedBackprop
from dapi.activations import project_layer_activations_to_input_rescale
from dapi.stereo_gc import get_sgc
from dapi.utils import normalize_image, image_to_tensor
import numpy as np
import scipy
import scipy.ndimage
import torch


def get_attribution(real_img,
                    fake_img,
                    real_class,
                    fake_class,
                    classifier,
                    methods=[
                        "ig",
                        "grads",
                        "gc",
                        "ggc",
                        "dl",
                        "ingrad",
                        "random",
                        "residual"
                    ],
                    bidirectional=False):

    '''Return (discriminative) attributions for an image pair.

    Args:

        real_img: (array-like)

            Real image to run attribution on.

        fake_img: (array-like)

            Counterfactual image typically created by a cycle GAN.

        real_class: (int)

            Class index of real image. Must correspond to networks output
            class.

        fake_class: (''int'')

            Class index of fake image. Must correspond to networks output
            class.

        classifier: (torch module)

            The classifier network to use.

        methods: (list of str)

            List of attribution methods to run. Possible values are: "ig"
            (Integrated Gradients), "grads" (just gradients), "gc" (GradCam),
            "ggc" (Guided GradCam), "dl" (DeepLift), "ingrad" (Input x
            Gradient), "random" (random attribution as a baseline), and
            "residual" (pixel-wise difference).

        bidirectional: (bool)

            Return both attribution directions.
    '''

    # get input shape and number of channels
    channels, height, width = real_img.shape
    input_shape = (height, width)

    imgs = [image_to_tensor(normalize_image(real_img).astype(np.float32)),
            image_to_tensor(normalize_image(fake_img).astype(np.float32))]

    classes = [real_class, fake_class]

    # prepare classifier
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    attrs = []
    attrs_names = []

    if "residual" in methods:
        res = np.abs(real_img - fake_img)
        res = res - np.min(res)
        attrs.append(torch.tensor(res/np.max(res)))
        attrs_names.append("residual")

    if "random" in methods:
        rand = np.abs(np.random.randn(*np.shape(real_img)))
        rand = np.abs(scipy.ndimage.filters.gaussian_filter(rand, 4))
        rand = rand - np.min(rand)
        rand = rand/np.max(np.abs(rand))
        attrs.append(torch.tensor(rand))
        attrs_names.append("random")

    if "gc" in methods:
        classifier.zero_grad()
        last_conv_layer = [
            (name, module)
            for name, module in classifier.named_modules()
            if type(module) == torch.nn.Conv2d
        ][-1]
        layer = last_conv_layer[1]
        layer_gc = LayerGradCam(classifier, layer)
        gc_real = layer_gc.attribute(imgs[0], target=classes[0])

        gc_real = project_layer_activations_to_input_rescale(
            gc_real.cpu().detach().numpy(),
            (input_shape[0], input_shape[1]))
        gc_real = np.stack([gc_real[0, 0, :, :],] * channels, axis=0)
        print(f"GC SHAPE {gc_real.shape}")

        attrs.append(torch.tensor(gc_real))
        attrs_names.append("gc")

        gc_diff_0, gc_diff_1 = get_sgc(
            real_img,
            fake_img,
            real_class,
            fake_class,
            classifier)
        attrs.append(gc_diff_0)
        attrs_names.append("d_gc")

        if bidirectional:
            gc_fake = layer_gc.attribute(imgs[1], target=classes[1])
            gc_fake = project_layer_activations_to_input_rescale(
                gc_fake.cpu().detach().numpy(),
                (input_shape[0], input_shape[1])
            )
            gc_fake = np.stack([gc_fake,] * channels, axis=0)

            attrs.append(torch.tensor(gc_fake))
            attrs_names.append("gc_fake")

            attrs.append(gc_diff_1)
            attrs_names.append("d_gc_inv")

    if "ggc" in methods:
        classifier.zero_grad()
        last_conv = [
            module
            for module in classifier.modules()
            if type(module) == torch.nn.Conv2d
        ][-1]

        # Real
        guided_gc = GuidedGradCam(classifier, last_conv)
        ggc_real = guided_gc.attribute(imgs[0], target=classes[0])
        attrs.append(ggc_real[0, :, :, :])
        print(f"GGC shape {ggc_real.shape}")
        attrs_names.append("ggc")

        gc_diff_0, gc_diff_1 = get_sgc(
            real_img,
            fake_img,
            real_class,
            fake_class,
            classifier)

        # D-gc
        classifier.zero_grad()
        gbp = GuidedBackprop(classifier)
        gbp_real = gbp.attribute(imgs[0], target=classes[0])
        ggc_diff_0 = gbp_real[0, :, :, :] * gc_diff_0
        attrs.append(ggc_diff_0)
        attrs_names.append("d_ggc")

        if bidirectional:
            ggc_fake = guided_gc.attribute(imgs[1], target=classes[1])
            attrs.append(ggc_fake[0, :, :, :])
            attrs_names.append("ggc_fake")

            gbp_fake = gbp.attribute(imgs[1], target=classes[1])
            ggc_diff_1 = gbp_fake[0, :, :, :] * gc_diff_1
            attrs.append(ggc_diff_1)
            attrs_names.append("d_ggc_inv")

    # IG
    if "ig" in methods:
        baseline = image_to_tensor(
            np.zeros(np.shape(real_img), dtype=np.float32)
        )
        classifier.zero_grad()
        ig = IntegratedGradients(classifier)
        ig_real, delta_real = ig.attribute(
            imgs[0],
            baseline,
            target=classes[0],
            return_convergence_delta=True)
        ig_diff_1, delta_diff = ig.attribute(
            imgs[1],
            imgs[0],
            target=classes[1],
            return_convergence_delta=True)

        attrs.append(ig_real[0, :, :, :])
        attrs_names.append("ig")

        attrs.append(ig_diff_1[0, :, :, :])
        attrs_names.append("d_ig")

        if bidirectional:
            ig_fake, delta_fake = ig.attribute(
                imgs[1],
                baseline,
                target=classes[1],
                return_convergence_delta=True)
            attrs.append(ig_fake[0, :, :, :])
            attrs_names.append("ig_fake")

            ig_diff_0, delta_diff = ig.attribute(
                imgs[0],
                imgs[1],
                target=classes[0],
                return_convergence_delta=True)
            attrs.append(ig_diff_0[0, :, :, :])
            attrs_names.append("d_ig_inv")

    # DL
    if "dl" in methods:
        classifier.zero_grad()
        dl = DeepLift(classifier)
        dl_real = dl.attribute(imgs[0], target=classes[0])
        dl_diff_1 = dl.attribute(imgs[1], baselines=imgs[0], target=classes[1])

        attrs.append(dl_real[0, :, :, :])
        attrs_names.append("dl")

        attrs.append(dl_diff_1[0, :, :, :])
        attrs_names.append("d_dl")

        if bidirectional:
            dl_fake = dl.attribute(imgs[1], target=classes[1])
            attrs.append(dl_fake[0, :, :, :])
            attrs_names.append("dl_fake")

            dl_diff_0 = dl.attribute(
                imgs[0],
                baselines=imgs[1],
                target=classes[0])
            attrs.append(dl_diff_0[0, :, :, :])
            attrs_names.append("d_dl_inv")

    # INGRAD
    if "ingrad" in methods:
        classifier.zero_grad()
        saliency = Saliency(classifier)
        grads_real = saliency.attribute(imgs[0],
                                        target=classes[0])
        grads_fake = saliency.attribute(imgs[1],
                                        target=classes[1])

        classifier.zero_grad()
        input_x_gradient = InputXGradient(classifier)
        ingrad_real = input_x_gradient.attribute(imgs[0], target=classes[0])

        ingrad_diff_0 = grads_fake * (imgs[0] - imgs[1])

        attrs.append(torch.abs(ingrad_real[0, :, :, :]))
        attrs_names.append("ingrad")

        attrs.append(torch.abs(ingrad_diff_0[0, :, :, :]))
        attrs_names.append("d_ingrad")

        if bidirectional:
            ingrad_fake = input_x_gradient.attribute(
                imgs[1],
                target=classes[1])
            attrs.append(torch.abs(ingrad_fake[0, :, :, :]))
            attrs_names.append("ingrad_fake")

            ingrad_diff_1 = grads_real * (imgs[1] - imgs[0])
            attrs.append(torch.abs(ingrad_diff_1[0, :, :, :]))
            attrs_names.append("d_ingrad_inv")

    attrs = [a.detach().cpu().numpy() for a in attrs]
    attrs_norm = [a/np.max(np.abs(a)) for a in attrs]

    return attrs_norm, attrs_names
