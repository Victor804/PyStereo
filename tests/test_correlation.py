import unittest
import numpy as np
import sys
sys.path.insert(1, '/home/victor/Documents/projets/stereoscopie/')
import correlation

class TestCorrelation(unittest.TestCase):
    def test_zncc(self):
        img1 = np.array([[1, 1, 0], [2, 10, 7], [11, 3, 0]])
        img2 = np.array([[1, 0, 0], [1, 9, 8], [12, 4, 1]])
        img3 = np.array([[7, 0, 3], [0, 5, 10], [5, 0, 2]])

        self.assertEqual(round(correlation.zncc(img1, img2), 2), 0.98)
        self.assertEqual(round(correlation.zncc(img1, img3), 2), 0.48)

    def test_ssd(self):
        img1 = np.array([[1, 1, 0], [2, 10, 7], [11, 3, 0]])
        img2 = np.array([[1, 0, 0], [1, 9, 8], [12, 4, 1]])
        img3 = np.array([[7, 0, 3], [0, 5, 10], [5, 0, 2]])

        self.assertEqual(round(correlation.ssd(img1, img2), 2), 7)
        self.assertEqual(round(correlation.ssd(img1, img3), 2), 133)

    def test_sad(self):
        img1 = np.array([[1, 1, 0], [2, 10, 7], [11, 3, 0]])
        img2 = np.array([[1, 0, 0], [1, 9, 8], [12, 4, 1]])
        img3 = np.array([[7, 0, 3], [0, 5, 10], [5, 0, 2]])

        self.assertEqual(round(correlation.sad(img1, img2), 2), 7)
        self.assertEqual(round(correlation.sad(img1, img3), 2), 31)


if __name__ == "__main__":
    unittest.main()
