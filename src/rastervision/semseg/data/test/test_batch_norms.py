import unittest

from rastervision.semseg.data.potsdam import (PotsdamNumpyFileGenerator,
                                              PotsdamImageFileGenerator)
from rastervision.common.utils import get_channel_stats
from rastervision.common.settings import datasets_path, TRAIN
PROCESSED_POTSDAM = 'isprs/processed_potsdam'


class NormalizationTestCase(unittest.TestCase):
    def test_numpy_potsdam_batch(self):
        generator = PotsdamNumpyFileGenerator(datasets_path, [0, 1, 2, 3, 4])
        gen = generator.make_split_generator(
            TRAIN, target_size=(10, 10), batch_size=100, shuffle=True,
            augment=False, normalize=True, only_xy=False)
        batch = next(gen)
        means, stds = get_channel_stats(batch.all_x)

        print(means, stds)
        # passes when mean = 0 with an error of +/- 0.3 and stds = 1 +/- 0.25.
        self.assertTrue((means > -.3).all() and (means < .3).all())
        self.assertTrue((stds > .75).all() and (stds < 1.25).all())

    def test_image_potsdam_batch(self):
        generator = PotsdamImageFileGenerator(datasets_path, [0, 1, 2, 3, 4])
        gen = generator.make_split_generator(
            TRAIN, target_size=(10, 10), batch_size=100, shuffle=True,
            augment=False, normalize=True, only_xy=False)
        batch = next(gen)
        means, stds = get_channel_stats(batch.all_x)
        print(means, stds)
        # passes when mean = 0 and stds = 1 with an error of +/- 0.25.
        self.assertTrue((means > -.25).all() and (means < .25).all())
        self.assertTrue((stds > .75).all() and (stds < 1.25).all())


if __name__ == '__main__':
    unittest.main()
