import numpy as np
import os

import cv2  # for resizing image

import sys

class BufferedImageSaver:
    """
    Stores incoming data in a Numpy ndarray and saves the array to disk once
    completely filled.
    """
    # rows = WINDOW_HEIGHT = 180
    # cols = WINDOW_HEIGHT = 320
    def __init__(self, filename: str, size: int,
                 rows: int, cols: int, depth:int, sensorname: str):
        """An array of shape (size, rows, cols, depth) is created to hold
        incoming images (this is the buffer). `filename` is where the buffer
        will be stored once full.
        """
        self.filename = filename + sensorname + '/'
        self.size = size
        self.sensorname = sensorname
        #dtype = np.float32 if self.sensorname == 'CameraDepth' else np.uint8
        
        ## dtype
        dtype = np.float32 if (self.sensorname == 'Lidar' or self.sensorname == 'Control' or self.sensorname == 'Control_real') else np.uint8
        
        self.buffer = np.zeros(shape=(size, rows, cols, depth),
                               dtype=dtype)
        self.index = 0
        self.reset_count = 0  # how many times this object has been reset
        
        ## check for syncrnization failures
        self.saved_count = 0

    def is_full(self):
        """A BufferedImageSaver is full when `self.index` is one less than
        `self.size`.
        """
        return self.index == self.size

    def reset(self):
        self.buffer = np.zeros_like(self.buffer)
        self.index = 0
        self.reset_count += 1

    def save(self):
        save_name = self.filename + str(self.reset_count) + '.npy'
        
        # make the enclosing directories if not already present
        folder = os.path.dirname(save_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
        self.saved_count += 1
        if(self.saved_count == self.reset_count + 1):
            # save the buffer
            np.save(save_name, self.buffer[:self.index + 1])
            print(self.size, " images saved in " + save_name)
            self.reset()
            return True
        else:
            print("frame skiped -----------------!!!!!!")
            #sys.exit(100)
            return False
            
    @staticmethod
    def process_by_type(img_bytes, name, buffer_rows, buffer_cols):
        """Converts the raw image to a more efficient processed version
        useful for training. The processing to be applied depends on the
        sensor name, passed as the second argument.
        """
        if name == 'CameraRGB':
            raw_img = np.frombuffer(img_bytes, dtype=np.uint8)
            #print(raw_img.shape)
            raw_img = raw_img.reshape(720, 1280, -1)
            #raw_img = np.reshape(raw_img, (720, 1280, 4))
            #raw_img = raw_img.reshape(720,1280,3)
            #raw_img = raw_img.reshape(buffer_rows, buffer_cols, -1)
            #print(buffer_rows, buffer_cols) # 180 320
            raw_img = raw_img[:, :, :3]
            #raw_img = np.resize(raw_img,(buffer_rows, buffer_cols, 3))
            raw_img = cv2.resize(raw_img, dsize=(buffer_cols, buffer_rows))
            #raw_img = raw_img[:, :, :3]
            
            return raw_img  # no need to do any processing

        elif name == 'CameraDepth':
            raw_img = np.frombuffer(img_bytes, dtype=np.uint8)
            raw_img = raw_img.astype(np.float32)
            total = raw_img[:, :, 2:3] + 256*raw_img[:, :, 1:2] + 65536*raw_img[:, :, 0:1]
            total /= 16777215
            return total
        
        elif name == 'CameraSemSeg':
            raw_img = np.frombuffer(img_bytes, dtype=np.uint8)
            return raw_img[:, :, 2: 3]  # only the red channel has information
            
        ## new: Lidar
        elif name == 'Lidar':
            points = np.frombuffer(img_bytes, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 1, 3))   # note: col is 1
            
            lidar_data = points #np.array(points[:, :, :2])
            
            lidar_rows = lidar_data.shape[0]
            
            if (lidar_rows < buffer_rows):
                #print("lidar_rows is less")
                ## use -1 as place holders for empty points
                #lidar_data.append()
                delta = buffer_rows - lidar_rows 
            elif (lidar_rows > buffer_rows):
                delta =  lidar_rows - buffer_rows 
                print("lidar_rows is more by", delta)
                
            return lidar_data
        
        ## Control signals
        elif name == "Control":
            control = img_bytes
            #print(control)
            control= np.reshape(control, (1, 1, 4))
            return control
            
        else:
            print("add_image saver name illegal")
            return None

    def add_image(self, img_bytes, name):
        """Save the current buffer to disk and reset the current object
        if the buffer is full, otherwise store the bytes of an image in
        self.buffer.
        """
        if self.is_full():
            success = self.save()
            if success:
                return self.add_image(img_bytes, name)
            else:
                return False
        else:
            #print('save')
            
            rows = self.buffer.shape[1]
            cols = self.buffer.shape[2]
            
            
            
            raw_image = self.process_by_type(img_bytes, name, rows, cols)
            
            #print(raw_image.shape)      #      (1888, 1, 3)
            #print(self.buffer.shape)    #(1000, 3000, 1, 3)
            
            
            # for Lidar: (0, 0, 0) will be the place holder in the buffer indicating an empty detection
            self.buffer[self.index][:raw_image.shape[0]] = raw_image
            #print(self.buffer[self.index])
            self.index += 1
            return True
            
            
            
            
            
            
            
            
