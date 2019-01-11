import os, argparse, numpy as np
from generate_det_file import generate_det_file
from YoloListVideo import getYoloDetNum
from shutil import copyfile

# ARRAY : Only those videos are exported
EXPORT = ['M0101', 'M0202', 'M0210', 'M0702', 'M1304']
# INTEGER : Discard 1 images over 2 (accelerate the video x2) = the step
SPEED_UP = 2
# INTEGER : Truncate videos which are too long
MAX_FRAMES = 500


parser = argparse.ArgumentParser(description='Convert the UAV benchmark dataset for the tracking step')
parser.add_argument('input_dataset', nargs='+', help='Path to the folder UAV-benchmark-M')
parser.add_argument('output', nargs='+', help='Path to the output directory')
args = parser.parse_args()


in_dir = args.input_dataset[0]
out_dir = args.output[0]


assert os.path.exists( out_dir ), "Output directory doesn't exists"

train_dir = os.path.join( out_dir, 'train' )
if not os.path.isdir( train_dir ):
	os.mkdir( train_dir )


for video in EXPORT:

	######################################################################################################################
	print("Exporting video frames")
	######################################################################################################################
	origin_dir = os.path.join(in_dir, 'images', video )
	export_dir = os.path.join(train_dir, video, 'img1', )
	os.makedirs( export_dir )

	images = [ i for i in os.listdir(origin_dir) if not i.startswith('.') ]
	images.sort()
	end = min( MAX_FRAMES, int( len(images) / SPEED_UP ) )
	for i in range(end):
		image = images[ i * SPEED_UP ]
		img_num = '0' * (6 - len(str(i + 1))) + str(i + 1)

		assert int(image[3:-4]) == i * SPEED_UP + 1, "Missing image... image {} has name {}".format(i*SPEED_UP+1, image)

		copyfile( os.path.join( origin_dir, image ), os.path.join( export_dir, img_num + ".jpg" ) )
	######################################################################################################################
	print("Exporting groundtruths")
	######################################################################################################################
	origin_dir = os.path.join(in_dir, 'gt' )
	export_dir = os.path.join(train_dir, video, 'gt' )
	os.mkdir( export_dir )


	gt_file = os.path.join(origin_dir, video + "_gt_whole.txt")
	assert os.path.exists( gt_file ), "The video {} has no groundtruths".format(video)
	gts = np.loadtxt( gt_file, int, delimiter=',', ndmin=1) # frame, id, left, top, width, height

	
	with open(os.path.join( export_dir, "gt.txt"), 'w' ) as f:
		# for each frame
		for i in np.arange(0, end * SPEED_UP, SPEED_UP):
			gt = gts[np.where(gts[:, 0] == i + 1)]
			# for each detection
			for entry in gt:
				#<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
				f.write( "{},{},{},{},{},{},1,-1,-1,-1\n".format(int(i/SPEED_UP) + 1, entry[1], entry[2], entry[3], entry[4], entry[5]) )
	######################################################################################################################
	#print("Exporting detections")
	######################################################################################################################
	#origin_dir = os.path.join(in_dir, 'det' )
	#export_dir = os.path.join(train_dir, video, 'det' )
	#os.mkdir( export_dir )

	#first, last = getYoloDetNum(os.path.join(in_dir, 'images'), video)

	#generate_det_file(origin_dir, os.path.join(export_dir, 'det.txt'), first, last, SPEED_UP)

