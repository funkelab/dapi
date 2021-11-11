from .base_options import BaseOptions

def downsample_type(s):
    try:
        factors = [tuple([int(x) for x in l.split(',')]) for l in s.split('x')]
    except:
        raise ValueError("Downsample input not understood")

    return factors


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--aux_checkpoint', type=str, help='AUX net checkpoint')
        parser.add_argument('--aux_input_size', type=int, default=128, help='AUX net input size')
        parser.add_argument('--aux_net', type=str, default='none', help='AUX net architecture')
        parser.add_argument('--aux_input_nc', type=int, default=1, help='AUX net input channels')
        parser.add_argument('--aux_output_classes', type=int, default=6, help='AUX number of output classes')
        parser.add_argument('--aux_downsample_factors', type=downsample_type, default=[(2,2), (2,2), (2,2), (2,2)], 
                            help='AUX Downsample factors for VGG')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
