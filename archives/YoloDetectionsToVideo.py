from python.helpers import *
import python.darknet as dn
import os, cv2, argparse

# this script must be runned from the darknet folder

input_dir   = "../video/"
output_path = "../output.mp4"

data        = "cfg/aerial.data"
cfg         = "cfg/yolov3-aerial.cfg"
weights     = "backup/yolov3-aerial.backup"


images = [ d for d in os.listdir( input_dir ) if d.endswith('.jpg') ]
images.sort()


img = cv2.imread( images[0] )
height, width, channels = img.shape

net = dn.load_net( cfg.encode(), weights.encode(), 0)
meta = dn.load_meta( data.encode() )

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

count = 0
total = len(images)
for i in images:
    count += 1
    print("Exportation {}%".format(count/total), end="\r")
    
    img_path = os.path.join(input_dir, i)
    r = dn.detect(net, meta, im.encode())
    out.write( draw_yolo_result(im, r, ['car', 'bus', 'truck'] ) )
    
out.release()
print("Exported {}".format(output_path))