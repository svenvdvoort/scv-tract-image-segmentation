import unittest
import scv_utility
from torch.utils.data import DataLoader

class TestExample(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(1 + 1, 2, "1 + 1 should equal 2")

    def test_dataset(self):
        dataset = scv_utility.MRIDataset('/Users/paul/Desktop/Projects/DeepComputerVision/uw-madison-gi-tract-image-segmentation')
        sample = dataset[0]

    def test_dataloader(self):
        dataset = scv_utility.MRIDataset(
            '/Users/paul/Desktop/Projects/DeepComputerVision/uw-madison-gi-tract-image-segmentation')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ == '__main__':
    unittest.main()
