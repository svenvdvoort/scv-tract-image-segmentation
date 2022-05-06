import unittest

class TestExample(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(1 + 1, 2, "1 + 1 should equal 2")

if __name__ == '__main__':
    unittest.main()
