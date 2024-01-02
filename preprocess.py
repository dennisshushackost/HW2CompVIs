import os
import numpy as np
from pathlib import Path
import albumentations as A

# TensorFlow and parameters to suppress warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import warnings

# Suppress warnings:
warnings.filterwarnings("ignore")

class Preprocessor(): 
    """
    This class is in charge of pre-processing the data and making 
    it suitable for the model.
    path: path to the directory containing the images and labels
    batch_size: batch size for training 
    num_classes: number of classes in the dataset (34 for this project)
    patch_size_width: width of the patches
    patch_size_height: height of the patches
    file_type_str: 'train', 'test', or 'validate'
    augment: True or False
    border: border size
    channels: number of channels (3 for RGB)     
    """

    def __init__(self,
                 path,
                 batch_size,
                 num_classes, 
                 patch_size_width, 
                 patch_size_height,
                 file_type_str,
                 augment=False,
                 border=0, 
                 channels=3
                 ):
        
        self.path = path
        self.batch_size = batch_size
        self.augment = augment
        self.border = border
        self.num_classes = num_classes
        self.patch_size_width = patch_size_width
        self.patch_size_height = patch_size_height
        self.file_type_str = file_type_str
        self.channels = channels
        self.dataset = None

        # Transform the images by applying augmentations (horizontal flip, vertical flip, rotation)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=40)
        ])

        # Get the list of files:
        path = Path(self.path)
        image_files = list(path.glob(self.file_type_str+'_img*.png'))
        label_files = list(path.glob(self.file_type_str+'_lbl*.png'))

        # Sort the files: 
        image_files.sort()
        label_files.sort()

        # Check that we have an equal amount of images and labels:
        if len(image_files) != len(label_files):
            raise ValueError('The number of images and labels does not match.')
        
        # Read the images and label file_directories into a numpy array:
        image_files_array = np.asarray([str(p) for p in image_files])
        label_files_array = np.asarray([str(p) for p in label_files])

        print(type(image_files_array))
        print(type(label_files_array))

        print("Number of images: ", len(image_files_array))
        print("Number of labels: ", len(label_files_array))


        # Shuffle the arrays without losing the order:
        combined = list(zip(image_files_array, label_files_array))
        np.random.shuffle(combined)
        image_files_array, label_files_array = zip(*combined)
        image_files_array = np.asarray(image_files_array)
        label_files_array = np.asarray(label_files_array)
   
        print("Here")
        # Read this into a tensorflow dataset:
        self.dataset = tf.data.Dataset.from_tensor_slices((image_files_array, label_files_array))

        # Show the first 10 files from the dataset:
        for image, label in self.dataset.take(10):
            print(image.numpy(), label.numpy())
 

 

        # Applies augmentations to the dataset:
        if self.augment and self.file_type_str=='train':
            self.dataset = self.dataset.map(lambda image, file: 
                        Preprocessor._parse_function(image, file, self.channels, self.augment_image_and_label))
        else:
            self.dataset = self.dataset.map(lambda image, file: 
                        Preprocessor._parse_function(image, file, self.channels))

    def _parse_function(self, image_filename, label_filename, augment_fn=None):
        """
        This function parses the image and label files into a tensor.
        It standardizes the images using min-max normalization.
        """

        # Read the image and label into a tensor:
        image_string = tf.io.read_file(image_filename)
        label_string = tf.io.read_file(label_filename)

        # Decode the image and label:
        image_decoded = tf.image.decode_png(image_string, channels=self.channels)
        label_decoded = tf.image.decode_png(label_string, channels=1)

        # Convert to float:
        image = tf.cast(image_decoded, tf.float32)
        label = tf.cast(label_decoded, tf.float32)

         # Apply augmentation:
        if augment_fn is not None:
            [image, label] = tf.numpy_function(augment_fn, [image, label], [tf.float32, tf.float32])
            image.set_shape([self.patch_size_width, self.patch_size_height, self.channels])
            label.set_shape([self.patch_size_width, self.patch_size_height, 1])

        # Min-max normalization:
        image_min = tf.reduce_min(image)
        image_max = tf.reduce_max(image)
        image = (image - image_min) / (image_max - image_min + 1e-9)

        label_string = tf.io.read_file(label_filename)
        label_decoded = tf.image.decode_png(label_string, channels=1)

        return image, label
    
    
    def augment_image_and_label(self,image,label):
        augmented = self.transform(image=image.numpy(), mask=label.numpy())
        return augmented['image'], augmented['mask']


# 
if __name__ == '__main__':
    dir_path = './data/train_small'
    batch_size = 32
    num_classes = 34 # Update with the actual number of your classes
    patch_size_width, patch_size_height = 256, 256  # Adjust sizes as needed
    file_type_str = 'train'  # Could be 'train', 'test', or 'validate'
    augment = False 
    border = 0
    channels = 3  # Typically 3 for RGB images

    # Create the preprocessor object:
    preprocessor = Preprocessor(dir_path, batch_size, num_classes, patch_size_width, patch_size_height, file_type_str, augment, border, channels)
    for image_batch, label_batch in preprocessor.dataset.take(1):
        # Do something with the batch, like display images or labels
        print(image_batch.shape, label_batch.shape)





    



