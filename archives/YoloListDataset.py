from DataManagerBenchmark import DataManager
import argparse, os

parser = argparse.ArgumentParser(description='List the images in the test/train sets')
parser.add_argument('attrs', nargs='+', help='Path to the folder containing the attributes')
parser.add_argument('vid_dir', nargs='+', help='Path to the folder containing the videos')

args = parser.parse_args()

videos_f = [ f for f in os.listdir( args.vid_dir[0] ) if not f.startswith('.') ]
videos_t = [ f[:5] for f in os.listdir( args.attrs[0] ) if f.endswith(".txt") ]

output = []

count = 0;
for video in videos_f:
	length = len( [ f for f in os.listdir( os.path.join(args.vid_dir[0], video) ) if f.endswith(".jpg") ] )
	if video in videos_t:
		output += range(count, count + length)
	count += length

for i in output:
	print( "/content/dataset/uav-yolo/{}.jpg".format(i) )