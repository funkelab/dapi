import argparse
import os

from dac.utils import open_image, save_image
from dac.attribute import get_attribution
from dac.mask import get_mask

parser = argparse.ArgumentParser()

parser.add_argument("--net", help="Name of network module in networks", required=True)
parser.add_argument("--checkpoint", help="Network checkpoint path", required=True)
parser.add_argument("--input_shape", help="Spatial image input shape", nargs="+", required=True)
parser.add_argument("--realimg", help="Path to real input image", required=True)
parser.add_argument("--fakeimg", help="Path to fake input image", required=True)
parser.add_argument("--realclass", help="Real class index", required=True, type=int)
parser.add_argument("--fakeclass", help="Fake class index", required=True, type=int)
parser.add_argument("--output_classes", help="Number of output classes", required=True, type=int)
parser.add_argument("--out", help="Output directory", required=False, default="dac_out")
parser.add_argument("--ig", help="Turn ON IG attr", action="store_true")
parser.add_argument("--grads", help="Turn ON grads attr", action="store_true")
parser.add_argument("--gc", help="Turn ON GC attr", action="store_true")
parser.add_argument("--ggc", help="Turn ON GGC attr", action="store_true")
parser.add_argument("--dl", help="Turn ON DL attr", action="store_true")
parser.add_argument("--ingrad", help="Turn ON ingrad attr", action="store_true")
parser.add_argument("--random", help="Turn ON random attr", action="store_true")
parser.add_argument("--residual", help="Turn ON random attr", action="store_true")
parser.add_argument("--downsample_factors", nargs="+", default=[2,2,2,2,2,2,2,2], help="Network downsample factors")



if __name__ == "__main__":
    args = parser.parse_args()
    input_shape = (int(args.input_shape[0]), int(args.input_shape[1]))
    real_img = open_image(args.realimg, flatten=True, normalize=False)
    fake_img = open_image(args.fakeimg, flatten=True, normalize=False)

    methods = []

    if args.ig:
        methods.append("ig")
    if args.grads:
        methods.append("grads")
    if args.gc:
        methods.append("gc")
    if args.ggc:
        methods.append("ggc")
    if args.dl:
        methods.append("dl")
    if args.ingrad:
        methods.append("ingrad")
    if args.random:
        methods.append("random")
    if args.residual:
        methods.append("residual")

    downsample_factors = [int(d) for d in args.downsample_factors]
    downsample_factors = [(downsample_factors[i], downsample_factors[i+1]) for i in range(0, len(downsample_factors),2)]

    if not methods:
        raise ValueError("Select at least one attribution method")

    mrf_scores = []
    mask_sizes = []

    # Fixed for now:
    channels = 1

    if args.net.lower() in ["vgg", "vgg2d"]:
        net = "Vgg2D"
    elif args.net.lower() in ["res", "resnet"]:
        net = "ResNet"

    attrs, attrs_names = get_attribution(real_img, fake_img,
                                         args.realclass, args.fakeclass,
                                         net, args.checkpoint,
                                         input_shape, channels, methods,
                                         output_classes=args.output_classes,
                                         downsample_factors=downsample_factors)


    for attr, name in zip(attrs, attrs_names):
        result_dict, img_names, imgs_all = get_mask(attr, real_img, fake_img, 
                                                    args.realclass, args.fakeclass, 
                                                    net, args.checkpoint, 
                                                    input_shape, channels,
                                                    args.output_classes, downsample_factors)

        method_dir = os.path.join(args.out, f"{name}")
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)

        k = 0
        for mask_imgs in imgs_all:
            threshold_dir = os.path.join(method_dir, f"t_{k}")
            if not os.path.exists(threshold_dir):
                os.makedirs(threshold_dir)
            for mask_im, mask_name in zip(mask_imgs, img_names):
                save_image(mask_im, os.path.join(threshold_dir, mask_name + ".png"))
            k += 1

        with open(os.path.join(method_dir, "results.txt"), 'w+') as f:
            print(result_dict, file=f)
