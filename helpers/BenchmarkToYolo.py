from DataManagerBenchmark import DataManager
import argparse

parser = argparse.ArgumentParser(description='Convert the UAV benchmark dataset for YoloV3 input')
parser.add_argument('input_img', nargs='+', help='Path to the folder containing the videos (each video being a folder of images)')
parser.add_argument('input_gt', nargs='+', help='Path to the folder containing the groundtruths .txt files')
parser.add_argument('output', nargs='+', help='Path to the output folder')


args = parser.parse_args()

def callback(a, b):
    print( "Exporting image {}/{} ({}%)".format(a, b, int( a*100 / b ) ), end="\r" )

print("Retrieving files ... ")
dataManager = DataManager(args.input_img[0], args.input_gt[0])

print("Exporting Dataset ...")
dataManager.export_dataset(args.output[0], False, 0, 1e5, callback)

print("\nDone.")