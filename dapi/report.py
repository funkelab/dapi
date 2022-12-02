from scipy.interpolate import interp1d
import numpy as np


class Report:

    def __init__(
            self,
            attribution,
            real,
            fake):

        self.attribution = attribution
        self.real = real
        self.fake = fake

        self.thresholds = []
        self.mask_sizes = []
        self.score_changes = []
        self.masks = []
        self.hybrids = []
        self.reals_masked = []
        self.fakes_masked = []
        self.diffs = []

        self.plot_mask_values = None
        self.plot_score_values = None

    def add_threshold(
            self,
            threshold,
            mask,
            mask_size,
            score_change,
            hybrid,
            real_masked,
            fake_masked,
            diff):

        self.thresholds.append(threshold)
        self.masks.append(mask)
        self.mask_sizes.append(mask_size)
        self.score_changes.append(score_change)
        self.hybrids.append(hybrid)
        self.reals_masked.append(real_masked)
        self.fakes_masked.append(fake_masked)
        self.diffs.append(diff)

        self.plot_mask_values = None
        self.plot_score_values = None

    def get_mask_and_score_values(self):
        """Returns the score change for equidistantly spaced mask sizes
        suitable for plotting and DAPI score computation.

        The mask size (x values in the plot) will be normalized between 0 (no
        mask) and 1 (whole image) and evenly spaced (0.01 units apart). The
        score change (y values in the plot) will be linearly interpolated from
        the ones provided by calling ``add_threshold``. The linear
        interpolation is necessary, since mask sizes for a sequence of
        thresholds do not increase linearly.

        Returns:

            (x_values, y_values), suitable for plotting and DAPI score
            computation.
        """

        if not self.plot_mask_values:

            image_size = np.prod(self.real.shape)

            normalized_mask_sizes = np.array(self.mask_sizes) / image_size
            score_changes = np.array(self.score_changes)

            f = interp1d(
                np.concatenate([[1], normalized_mask_sizes, [0]]),
                np.concatenate([[score_changes[0]], score_changes, [0]]))
            self.plot_mask_values = np.arange(0.0, 1.0001, 0.01)
            self.plot_score_values = [f(x) for x in self.plot_mask_values]

        return self.plot_mask_values, self.plot_score_values

    def get_dapi_score(self):
        """Returns the DAPI score for all thresholds of the attribution map
        added by ``add_threshold``.

        The DAPI score is the area under the mask-size vs. score-change curve.
        """

        x_values, y_values = self.get_mask_and_score_values()

        # DAPI score = AUC of above x-y values
        dapi_score = np.trapz(y_values, x_values)

        return dapi_score
