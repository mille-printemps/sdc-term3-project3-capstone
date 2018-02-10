from styx_msgs.msg import TrafficLight

import rospy
import tensorflow as tf
from keras.utils.data_utils import get_file
import os
import numpy as np
import time
import errno

MODEL_FILENAME = 'frozen_inference_graph.pb'
SCORE_THRESHOLD = 0.3

class TLClassifier(object):
    def __init__(self, consensus=1):
        data = rospy.get_param('~data')

        self.detector_model_path =  '../../../data/models/ssd/' + '/'.join([data, MODEL_FILENAME])
        self.detector = Detector(self.detector_model_path)
        self.consensus = consensus

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        now = rospy.get_time()

        (boxes, scores, classes, num_detections) = self.detector.detect(image)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        rospy.logdebug("scores = {}, classes = {}".format(scores, classes))

        num_classes = np.zeros(5, dtype=np.uint8)
        sum_scores = np.zeros(5)
        for i in range(len(classes)):
            color = self.eval_color(classes[i])
            score = scores[i]
            if SCORE_THRESHOLD < score:
                num_classes[color] += 1
                sum_scores[color] += score

        rospy.loginfo("num_classes = {}, sum_scores = {}".format(num_classes, sum_scores))

        prediction = -1
        maxn = np.max(num_classes)
        if self.consensus <= maxn:
            cands = (num_classes == maxn)
            prediction = np.argmax(sum_scores * cands)

        rospy.loginfo("prediction = {}, max_score = {}".format(prediction, scores[0]))
        rospy.loginfo("prediction time = {:.4f} s".format(rospy.get_time() - now))

        return prediction        

    def eval_color(self, classification):
        if classification == 1:
            return TrafficLight.RED
        elif classification == 2:
            return TrafficLight.YELLOW
        elif classification == 3:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN

class Detector(object):
    def __init__(self, model_file):
        try:
            os.makedirs(os.path.dirname(model_file))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.model_file = os.path.abspath(model_file)
        self.detection_graph = tf.Graph()
        self.import_tf_graph()
        self.session = tf.Session(graph=self.detection_graph)

    def import_tf_graph(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def detect(self, image_np):
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        return self.session.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np})
