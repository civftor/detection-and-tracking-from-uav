import numpy as np
import tensorflow as tf

import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from utils import label_map_util
from utils import visualization_utils as vis_util

def detect(model_dir, images_path):
  return infer( prepare_det(model_dir), model_dir, images_path )

def prepare_det(model_dir):

  PATH_TO_GRAPH = os.path.join(model_dir, 'frozen_inference_graph.pb')
  PATH_TO_LABELS = '/avd/tensorflow_configs/benchmark_label_map.pbtxt'
  NUM_CLASSES = 7

  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')
      sess = tf.Session(graph=detection_graph)
  return detection_graph, category_index, sess

def infer(pack, model_dir, images):
  detection_graph, category_index, sess = pack
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')

  boxes = []

  for image in images:
    image = cv2.imread(path)
    image_expanded = np.expand_dims(image, axis=0)
    detections = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    boxes.append( detections )

  return boxes



