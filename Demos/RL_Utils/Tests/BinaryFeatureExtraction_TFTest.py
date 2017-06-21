import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers.BinaryFeatureExtraction_TF import BinaryFeatureExtractor

class BinaryFeatureExtractor_TEST(tf.test.TestCase):

  def testSquare(self):
    with self.test_session():
      bfe = BinaryFeatureExtractor([21,3])
      y = bfe.activateBinaryFeatures_TF(10)
      y_pythonic = bfe.activateBinaryFeaturesBrute(10)
      self.assertAllEqual(y.eval(), np.transpose([y_pythonic]))

if __name__ == '__main__':
  tf.test.main()
  
  
  