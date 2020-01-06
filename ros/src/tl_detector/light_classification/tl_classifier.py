#!/usr/bin/env python
import os
import rospy
import rospkg
import sys
import time
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight

def load_graph(graph_file):
    """
    Loads the frozen inference protobuf file
    """
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='prefix')
    return graph


def filter_results(min_score, scores, classes):
    """
    Returns tuple (scores, classes) where scores[i] >= `min_score`
    """
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)

    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_scores, filtered_classes


class TLClassifier(object):

    def __init__(self, is_site=False):
        #TODO load classifier
        self.__model_loaded = False
        self.tf_session = None
        self.prediction = None
        self.path_to_model = '../../../deep_learning/models/frozen_graphs/'

        self.load_model(is_site)

    def load_model(self, is_site):
        detect_path = rospkg.RosPack().get_path('tl_detector')
        if is_site:
           self.path_to_model += 'real_mobilenets_ssd_38k_epochs_frozen_inference_graph.pb'
        else:
           self.path_to_model += 'sim_mobilenets_ssd_30k_epochs_frozen_inference_graph.pb'
        rospy.loginfo('Load model from ' + self.path_to_model)

        # Load the graph using the path
        self.tf_graph = load_graph(self.path_to_model)

        # Configurations below
        self.config = tf.ConfigProto(log_device_placement=False)
        # GPU video memory usage setup
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # Setup timeout for any inactive option
        self.config.operation_timeout_in_ms = 50000

        # Placeholders below
        # Image
        self.image_tensor = self.tf_graph.get_tensor_by_name('prefix/image_tensor:0')
        # Number of predictions found in the image
        self.num_detections = self.tf_graph.get_tensor_by_name('prefix/num_detections:0')
        # Confidence of the prediction
        self.detection_scores = self.tf_graph.get_tensor_by_name('prefix/detection_scores:0')
        # Classification of the prediction (integer id)
        self.detection_classes = self.tf_graph.get_tensor_by_name('prefix/detection_classes:0')

        with self.tf_graph.as_default():
            self.tf_session = tf.Session(graph=self.tf_graph, config=self.config)

        self.__model_loaded = True
        rospy.loginfo("Successfully loaded model and configured placeholders")


    def get_classification(self, image, confidence_cutoff=0.3):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if not self.__model_loaded:
            return TrafficLight.UNKNOWN

        colors = ["RED", "YELLOW", "GREEN"]
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        # Get the scores, classes and number of detections
        # Re-use the session: it is more than 3x faster than creating a new one for every image
        (scores, classes, num) = self.tf_session.run(
            [self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np})

        # Remove unecessary dimensions
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # And prune results below the cutoff
        final_scores, final_classes = filter_results(confidence_cutoff, scores, classes)

        if len(final_classes) == 0:
            rospy.loginfo("[WARN] Predicted color does not make the cut " + colors[classes[0] - 1])
            return TrafficLight.UNKNOWN

        # TrafficLight messages have red = 0, yellow = 1, green = 2.
        # The model is trained to identify class red = 1, yellow = 2, green = 3.
        # Hence, subtracting 1 to match with TrafficLight message spec.
        rospy.loginfo("Predicted color is " + colors[final_classes[0] - 1])
        return final_classes[0] - 1
