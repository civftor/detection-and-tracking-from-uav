import tensorflow as tf
import numpy as np
import os
import contextlib2
import random
import argparse
from glob import glob
from shutil import copyfile
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from random import shuffle
from PIL import Image


parser = argparse.ArgumentParser(description='Convert the UAV benchmark dataset for tensorflow input (sharded tf_records)')
parser.add_argument('input', nargs='+', help='Path to the folder containing the folders "images", "test", "train" and "gt"')
parser.add_argument('output', nargs='+', help='Path to the output folder')

args = parser.parse_args()



def create_tf_example(example):
  filepath, labels = example
  
  width = 1024
  height = 540
  
  fp = open(filepath, 'rb')
  image = fp.read()
  
  filename = (os.path.basename(os.path.dirname(filepath)) + "/" + os.path.basename(filepath)).encode() # Filename of the image. Empty if image is not from file
  
  encoded_image_data = image                     # Encoded image bytes
  image_format = b'jpeg'                         # b'jpeg' or b'png'
  
  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
  
  classes = []      # List of integer class id of bounding box (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  
  classes_names = [b'', b'car', b'truck', b'bus']
  
  for label in labels:
    xmins.append( label[2] / width )
    ymins.append( label[3] / height )
    xmaxs.append( (label[2] + label[4]) / width )
    ymaxs.append( (label[3] + label[5]) / height )
    classes.append( int(label[8]) )
    classes_text.append( classes_names[int(label[8])] )

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def shard_tf( output_filebase, examples, num_shards=10 ):
  count = len(examples)
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
    for index, example in enumerate(examples):
      print("Creating example {}/{}".format(index, count), end="\r")

      tf_example = create_tf_example(example)
      output_shard_index = index % num_shards
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

def create_tf( input_name, num_shards = 1, subset = -1 ):
  """input_name : "train" or "test"
  """

  input_folder = args.input[0];
  output_folder = args.output[0];
  
  videos = glob( os.path.join(input_folder, input_name, "*_attr.txt") )

  examples = []
  for video in videos:
    video_name = os.path.basename(video)[:5]
    label_path = os.path.join(input_folder, "gt", video_name + "_gt_whole.txt")
    assert os.path.exists(label_path), "The expected path to groundtruth files does not exists : " + label_path

    video_label = np.loadtxt( label_path, delimiter=",")
    video_frames = glob( os.path.join(input_folder, "images", video_name, "*.jpg") )
    
    if subset > 0:
      video_frames = random.sample(video_frames, subset)
    video_frames.sort()
    
    for frame in video_frames:
      frame_index = int(frame[-10:-4])
      label = video_label[ np.where( video_label[:,0] == frame_index ) ]
      examples.append( (frame, label) )

  random.shuffle(examples)

  output_path = os.path.join(output_folder, input_name + ".record")
  
  print("Exporting the dataset ... ")
  shard_tf( output_path, examples, num_shards )
  print("\nOutput : {}".format(output_path))


create_tf("train", 12) #, 250)
create_tf("test", 4) #, 25)




