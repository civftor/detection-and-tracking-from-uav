import tensorflow as tf
import numpy as np
import os
import sys
import warnings
import contextlib2
import random
from glob import glob

currentDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append( os.path.join(currentDir, "..", "tensorflow") )

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from PIL import Image

#
# Create sharded Tensorflow records
# 
# The dataset is sharded into num_shards records file
# PNG files are reformated as jpegs
# Images whose width or height exceed 1024 pixels are resized
# 


#### Paths to the aerial dataset and the output ####
input_folder = "/Volumes/Disque-Dur-S-C/aerial_dataset/";
output_path = "/Volumes/Disque-Dur-S-C/aerial_tf_records_jpg/train.record";
num_shards = 8
MAX_SIZE = 1024


def resize_image(im):
    tmp_filepath = ".rec-export-tmp.jpg"

    # Keep aspect ratio
    im.thumbnail( (MAX_SIZE, MAX_SIZE) )

    # Todo : save directly to a byteArray without accessing hard drive.
    im.save(tmp_filepath, "JPEG", optimize=True)
    fp = open(tmp_filepath, 'rb')
    byteArray = fp.read()
    os.remove(tmp_filepath)

    return byteArray

def bound( i ):
  return min( max( i, 0 ), 1 )

def create_tf_example(example):
  filepath, boxes = example

  # Note : Following 2 lines don't read the image data
  im = Image.open(filepath)
  width, height = im.size
  
  if width > MAX_SIZE or height > MAX_SIZE or not os.path.splitext(filepath)[1] == ".jpg":
    image = resize_image(im)
  else:
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
  
  classes_names = [b'', b'1', b'2', b'3', b'4', b'5', b'6']

  # To do : refactor using numpy to parallelize ?
  for box in boxes:
    #class_no  xc yc w h
    xmins.append( bound(box[1] - box[3]/2) )
    ymins.append( bound(box[2] - box[4]/2) )
    xmaxs.append( bound(box[1] + box[3]/2) )
    ymaxs.append( bound(box[2] + box[4]/2) )
    _c = int(box[0]) + 1
    classes.append( _c )
    classes_text.append( classes_names[ _c ] )

    if not ( xmins[-1] < xmaxs[-1] and ymins[-1] < ymaxs[-1] ):
      print("Warning : negative width or height. Box skipped. (x1, y1, x2, y2) {}, {}, {}, {}".format(xmins[-1], xmaxs[-1], ymins[-1], ymaxs[-1]) )
      continue

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

def create_sharded_tf_record( output_filebase, examples, num_shards=10 ):
  count = len(examples)
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
    for index, example in enumerate(examples):
      print("Creating example {}/{}".format(index, count), end="\r")

      tf_example = create_tf_example(example)
      output_shard_index = index % num_shards
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())



#### List all images and load labels ####

print("Loading dataset from {}".format(input_folder))
examples = []

# Foreach subfolder
datasets = [ds for ds in os.listdir(input_folder) if os.path.isdir( os.path.join(input_folder, ds) )]
_c = 0
for ds in datasets:
  print("{}/{}".format( _c,len(datasets) ), end='\r')
  _c += 1

  images_path = glob( os.path.join(input_folder, ds, "*.jpg") )
  images_path.extend(glob( os.path.join(input_folder, ds, "*.png") ))

  # Open files in the sorted order may diminue the hard drive pointer movements
  images_path.sort()
  
  # Foreach image in this subfolder
  for image_path in images_path:
    label_path = os.path.splitext(image_path)[0] + ".txt"
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      label = np.loadtxt( label_path, delimiter=" ", ndmin=2 )
      examples.append( (image_path, label) )


random.shuffle(examples)
print("Exporting the dataset containing {} entries.".format(len(examples)))
create_sharded_tf_record( output_path, examples, num_shards )
print("Done\nOutput : {}".format(output_path))


