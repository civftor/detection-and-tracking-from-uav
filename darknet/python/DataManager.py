import cv2
import numpy as np
import os, sys, random


class DataManager:
    
    # 16 / 9 Format
    OUTPUT_WIDTH = 1024 #640
    OUTPUT_HEIGHT = 576 #360

    # dataset
    #   videos_path
    #   labels_path
    #
    #   videos = ["uav0", ...]
    #   frames = [ ["0.jpg", ...], ... ]
    #   labels = ["path/uav0.txt", ...]
    #
    # video_count = 0
    # frame_counts = []
    #
    # loaded_label = np.Array(int)
    # loaded_label_index = 0
    #
    # current_dir
    # current_frame

    def __init__(self, videos_path, labels_path, subset = -1):
        self._dataset = self.__scan_paths( videos_path, labels_path, subset )
        
        self._video_count = len( self._dataset['videos'] )
        self._frame_counts = [ len( d ) for d in self._dataset['frames'] ]
        self._total_count = sum( self._frame_counts )
        
        self.reset()

        print( "The dataset contains %d frames " % self._total_count )

    def reset(self, start = 0):
      
        a = 0
        b = start
        while b > self._frame_counts[a]:
          b -= self._frame_counts[a]
          a += 1
          
        self._current_dir = a
        self._current_frame = b
        self._loaded_label_index = -1
        
    def has_next(self):
        return  self._current_dir < self._video_count
    
    def get_next(self, resize = False):
      
        img = self.__get_img_at( self._current_dir, self._current_frame )
        label = self.__get_label_at(  self._current_dir, self._current_frame )

        # Go to next frame if any or first frame of next video
        if self._current_frame + 1 < self._frame_counts[self._current_dir]:
            self._current_frame += 1
        else:
            self._current_dir += 1
            self._current_frame = 0

        if resize:
            return self.__resize( img, label )
        else:
            return img, label
          
      
    def export_dataset(self, destination_path, label_only = True, start = 0, end = 1e10, update_callback = False):

        if os.path.exists(destination_path):
            if start == 0:
              if not os.path.isdir(destination_path):
                  raise Exception('Destination path is a file')
              if os.listdir(destination_path):
                  raise Exception('Destination directory already exists and is not empty')
        else:
            os.mkdir( destination_path )

        if start == -1:
            start = 0
        
        self.reset(start)

        output_count = start
        output_size = min( self._total_count, end )

        while self.has_next() and output_count < end:
            img, label = self.get_next()
            
            dest = os.path.join(destination_path, str(output_count))
            
            
            self.export_label( dest + ".txt", img, label)
            if not label_only:
                self.export_img( dest + ".jpg", img, label)
            
            output_count += 1

            if update_callback:
                update_callback( output_count, output_size )
            
    def export_img(self, dest, img, label):
      
        img = cv2.resize(img, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
        cv2.imwrite(dest, img)
      
    def export_label(self, dest, img, label):
        #######
        #
        # input format
        # | frame_index | object_id | bbox_x | bbox_y | bbox_width | bbox_height | confidence | category | truncation | occlusion |
        #
        # output format
        # | class_number | center_x / image_width | center_y / image_height | absolute_width / image_width | absolute_height / image_height |
        #
        ########
        #
        # input classes
        # | car | truck | ... |
        #
        # output classes
        # | car | truck | ... |
        #
        ########
        
        image_width = img.shape[1]
        image_height = img.shape[0]
        
        
        with open(dest, 'w') as file:
          for box in label:
              new_box = []

              # Export class index
              new_box.append( box[7] )

              # Export relative box center
              center_x = box[2] + box[4] / 2
              center_y = box[3] + box[5] / 2
              new_box.append( center_x / image_width )
              new_box.append( center_y / image_height )

              # Export relative box size
              new_box.append( box[4] / image_width )
              new_box.append( box[5] / image_height )

              file.write( ' '.join( str(x) for x in new_box) + "\n" )
      

    def __scan_paths( self, videos_path, labels_path, subset ):
        assert os.path.exists( videos_path ), "Data path does not exists: " + data_path
        assert os.path.exists( labels_path ), "Labels path does not exists: " + label_path
        
        dataset = {
            'videos_path': videos_path,
            'labels_path': labels_path,
            'videos': [],  # List of videos directories
            'labels': [],  # List of labels filename
            'frames': []   # List of frames filename
        }

        dataset['videos'] = [ d for d in os.listdir( videos_path ) if not d.startswith('.') ]
        dataset['videos'].sort()

        for i, d in enumerate( dataset['videos'] ):            
            dataset['frames'].append( [ f for f in os.listdir( os.path.join( videos_path, d ) ) if not f.startswith('.') ] )
            if subset > 0:
                dataset['frames'][-1] = random.sample(dataset['frames'][-1], subset)
            dataset['frames'][-1].sort()
            dataset['labels'].append( os.path.join( labels_path, d + ".txt" ) )

        return dataset

    def __get_img_at(self, i, j):
        
        video = self._dataset['videos'][i]
        frame = self._dataset['frames'][i][j]

        path = os.path.join( self._dataset['videos_path'], video, frame)

        return self.__load_image( path )

    def __get_label_at(self, i, j):

        if self._loaded_label_index != i:
            self._loaded_label = self.__load_label( self._dataset['labels'][i] )
            self._loaded_label_index = i

        index = int( self._dataset['frames'][i][j][:7] )
        return self._loaded_label[np.where(self._loaded_label[:, 0] == index)];

    def __load_label(self, filename):
        return np.loadtxt(filename, int, delimiter=",")
        
    def __load_image(self, filename):
        return cv2.imread(filename)

    def __resize(self, img, label):

        x_ratio = self.OUTPUT_WIDTH / img.shape[1]
        y_ratio = self.OUTPUT_HEIGHT / img.shape[0]

        img = cv2.resize(img, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))

        label[:, [2, 4]] = (label[:, [2, 4]] * x_ratio).astype(int)
        label[:, [3, 5]] = (label[:, [3, 5]] * y_ratio).astype(int)

        return img, label