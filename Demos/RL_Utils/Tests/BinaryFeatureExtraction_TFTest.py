import tensorflow as tf
from BinaryFeatureExtraction_TF import BinaryFeatureExtractor

class BinaryFeatureExtractor_TEST(tf.test.TestCase):

  def testSquare(self):
    with self.test_session():
      bfe = BinaryFeatureExtractor()
      bfe.initBinaryFeatureList(21, 3)
      y = bfe.activateBinaryFeatures_TF(10)
      y_pythonic = bfe.activateBinaryFeaturesBrute(10)
      self.assertAllEqual(y.eval(), y_pythonic)

if __name__ == '__main__':
  tf.test.main()
  
  
  