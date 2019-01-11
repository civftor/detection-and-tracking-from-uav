import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os
import errno

def directory_to_video( input_images, output_dir ):
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    output_path = os.path.join(output_dir, "output.mp4")

    # Determine the width and height from the first image
    img = cv2.imread( input_images[0] )
    height, width, channels = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for i in input_images:
        img = cv2.imread(i)
        out.write(img)

    out.release()
    return output_path


def draw_yolo_result(im, objects, names = ['car', 'truck', 'bus', 'minibus', 'cyclist']):
    cmap = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48]]

    img = cv2.imread(im);

    for obj in objects:
        topleft  = ( int(obj[2][0] - obj[2][2]/2), int(obj[2][1] - obj[2][3]/2) )
        botright = ( int(obj[2][0] + obj[2][2]/2), int(obj[2][1] + obj[2][3]/2) )
        textpos  = (topleft[0], topleft[1] - 7)

        c = names.index(obj[0].decode())
        text = obj[0].decode() + "=" + str(round(obj[1], 2))

        cv2.rectangle(img, topleft, botright, cmap[c], 2)
        cv2.putText(img, text , textpos, 2, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def get_classes_name():
    return ['regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']

def get_classes_colors():
    return [[230, 25, 75], [60, 180, 75], [255, 225, 25],
            [0, 130, 200], [245, 130, 48], [145, 30, 180],
            [70, 240, 240], [240, 50, 230], [210, 245, 60],
            [250, 190, 190], [0, 128, 128], [230, 190, 255]]

def read_frame(path, frame):
    path = os.path.join(path, str(frame))
    return cv2.imread( path + ".jpg" ), np.loadtxt(path + ".txt")

def draw_objects_on_img(img, label):
    cmap = get_classes_colors()
    names = get_classes_name()

    for obj in label:

        width = img.shape[1]
        height = img.shape[0]

        topleft  = ( int( (obj[1] - obj[3]/2) * width ), int( (obj[2] - obj[4]/2) * height ) )
        botright = ( int( (obj[1] + obj[3]/2) * width ), int( (obj[2] + obj[4]/2) * height ) )
        textpos  = (topleft[0], topleft[1] - 7)

        c = int( obj[0] )

        cv2.rectangle(img, topleft, botright, cmap[c], 2)

        cv2.putText(img, names[c] , textpos, 2, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img

def display_img( img ):
    plt.figure( figsize=(15, 10) )
    plt.imshow( img )

def export_to_video( dm, output_dir ):

    raise Exception("Not implemented")

    progress_bar = IntProgress(min = 0, max = dm.get_length())
    display(progress_bar)


    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    output_path = os.path.join(output_dir, "output.mp4")

    # Determine the width and height from the first image
    img, label = dm.get_next()
    height, width, channels = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    out.write( draw_objects_on_img(img, label) )

    while( dm.has_next() ):
        img, label = dm.get_next()
        img = draw_objects_on_img(img, label)

        out.write(img) # Write out frame to video
        progress_bar.value += 1

    # Release everything if job is finished
    out.release()

    return output_path

def symlink(src, dst):
    """
    Create a simlink at dst which points to src
    """
    try:
        os.symlink(src, dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(dst)
            os.symlink(src, dst)







