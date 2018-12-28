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

class DataTransformation:
	_loaded = false
	_loaded_mode = None
	_imagesPath = []
	_labelsPath = []


	MODE_YOLO = "yolo"
	MODE_UAVDT = "uavdt"

	def scanVideosFolder( self, path ):
		"""
			return the list of (only) folders at the root of "path"
		"""
		folders = os.listdir( path )
		ret = []

		for f in folders:
			f2 = os.path.join(path, f)
			if not f.startswith(".") and os.path.isdir( f2 ):
				ret.append(f2)

		return ret


	def scanImagesFolder( self, path, exts = [".jpg", ".png"] ):
		"""
			return the list of "exts" images at the root of "path"
		"""
		images = os.listdir(path)
		ret = []

		for i in images:
			if not i.startswith("."):
				for ext in exts:
					if i.endswith( ext ):
						ret.append( os.path.join(path, i) )

		return ret

	def yoloInput( self, image_path_list ):
		"""
			infering label path :
				path/to/image001.jpg -> path/to/image001.txt
		"""
		assert len(image_path_list) > 0, "Empty input list"

		_loaded_mode = MODE_YOLO

		_imagesPath = image_path_list

		for path in _imagesPath:
			label_path = os.path.splitext(path)[0] + ".txt"
			label = np.loadtxt( label_path, delimiter=" " )


	def uavInput( self, root_dir ):
		"""
			root_dir is the folder containing the videos (one folder of images per video)
		"""

		videos = self.scanVideosFolder( root_dir )
		










