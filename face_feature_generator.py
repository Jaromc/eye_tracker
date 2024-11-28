import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from os.path import exists
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import cv2
import albumentations as A
from keras import backend as K
import time
import os

dir_300w_dataset = ""
dir_UTK_image_folder = ""
dir_UTK_csv_path = ""
dir_model_save_folder = ""
dir_debug_output_folder = ""
dir_results_output_folder = ""

def print_model_image(img, points, i, batch):
   it = iter(points)
   fig, ax = plt.subplots()
   ax.imshow(img)
   for x in it:
      y = next(it)
      ax.plot(x, y,'ro')
   plt.savefig(dir_debug_output_folder+ str(batch) + "_" + str(i) + ".png")
   plt.clf()
   plt.close()
 
def augment_images(images, landmarks, composed_transform, landmark_count):
  newimages = []
  newlandmarks = []
  print(len(images))
  for i in range(0,len(images)):
      image = images[i]
      #reshape into xy format. -1 infers the array size from the input dimension
      xy_points = np.reshape(landmarks[i], (-1, 2))
      transformed = composed_transform(image=image, keypoints=xy_points)

      #uncomment to see transformed image
      #print_image(transformed['image'], transformed['keypoints'], i)

      #ShiftScaleRotate can move the keypoints so we end up with a different amount
      #than we expect. We are lazy and just throw these results away. 
      if len(transformed['keypoints']) != landmark_count/2:
          continue

      newimages.append(transformed['image'])
      #convert points back into 1d array
      newlandmarks.append(np.reshape(transformed['keypoints'], -1))

      #uncomment to confirm format is back to how our model expects it
      #print_model_image(images[i], landmarks[i], i, 0)

  return np.array(newimages, dtype='float32'), np.array(newlandmarks, dtype='int')

class My_Custom_Generator(tf.keras.utils.Sequence) :
  
  def __init__(self, images, landmarks, batch_size) :
    self.batch_size = batch_size

    self.images = images
    self.landmarks = landmarks
    self.augmentation_count = 0

    self.augmented_images, self.augmented_landmarks = self.createNewSet(self.images, self.landmarks, self.images.shape[0])
    random.seed(74)

  def createNewSet(self, images, landmarks_sample, target_length):

    new_images = None
    new_landmarks = None
    affine_transform = A.Compose(
        #numbers chosen to move an image and landmarks mostly within our original bounds.
        [A.Affine(scale=(0.5,1.3), translate_percent=(-0.4, 0.4), rotate=(-30,30), cval=0, mode=cv2.BORDER_CONSTANT, keep_ratio=True, p=1)],
        keypoint_params=A.KeypointParams(format='xy') #consider param 'remove_invisible'
    )

    # create augmented data till we have the same number we began with. Sometimes with the random
    # augmentations we need to throw some away if the landmarks arn't useable
    while True:
        # The dataset has landmarks for the whole face. We only care about the 12 points around the eyes.
        # We pass in 24 because each point is x,y == 12*2.
        affine_images, affine_landmarks_sample = augment_images(images, landmarks_sample, affine_transform, 24)

        if new_images is None:
           new_images = affine_images
           new_landmarks = affine_landmarks_sample
        else:
            new_images = np.concatenate([new_images, affine_images])
            new_landmarks = np.concatenate([new_landmarks, affine_landmarks_sample])

        #cut final list to length if needed and return when we have reached our target size
        if new_images.shape[0] == target_length:
            break
        elif new_images.shape[0] > target_length:
            new_images = new_images[:target_length]
            new_landmarks = new_landmarks[:target_length]
            break

    print("New dataset created")

    # Uncomment to debug augmentations
    # print("Printing images")
    # for i in range(0, len(new_images)):
    #    print_model_image(new_images[i], new_landmarks[i], i, self.augmentation_count)
    # print("Printing images complete")

    self.augmentation_count += 1

    return new_images, new_landmarks

  def on_epoch_end(self):
    #create a new augmented dataset each epoch
    self.augmented_images, self.augmented_landmarks = self.createNewSet(self.images, self.landmarks, self.images.shape[0])
    
  def __len__(self) :
    len = int(np.floor((self.augmented_images.shape[0] / self.batch_size)))
    return len
  
  def __getitem__(self, idx) :
    #get batches from our dataset for the next epoch
    batch_x = self.augmented_images[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.augmented_landmarks[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return batch_x, batch_y

class stopOnLoss(tf.keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs={}):
      if(logs.get('loss') <= 0.1):
         print("\n\n\nReached defined loss value so cancelling training!\n\n\n")
         self.model.stop_training = True

def loadUTKSet():
    raw_data = pd.read_csv(dir_UTK_csv_path, header=None, delim_whitespace=True)

    #create column names. This isn't necessary for this application
    columns_names = ['filename']
    convert_dict = {'filename': str}
    for idx in range(0, raw_data.shape[1]-1):
        columns_names.append(str(idx))
        convert_dict[str(idx)] = int

    raw_data.columns = columns_names
    raw_data = raw_data.astype(convert_dict)

    # idx_column_names = columns_names[1:len(columns_names)]
    # print(len(columns_names))
    # print(len(idx_column_names))
    # print(idx_column_names)

    image_folder = dir_UTK_image_folder
    existing_data = raw_data[[exists(image_folder + i) for i in raw_data[raw_data.columns[0]]]]
    print("Raw data shape")
    print(existing_data.shape)

    eye_data = existing_data.iloc[:,73:97]#existing_data.iloc[:,73:97] #73:97 is eye data only. 1:137 is whole face
    print(eye_data.shape)
    filenames = existing_data.iloc[:,0]
    landmarks = eye_data

    #Grab only a portion of the dataset. This can be removed if you have the memory to handle it
    filenames_sample = filenames[3000:18000]
    landmarks_sample = landmarks[3000:18000]
    print(filenames_sample.shape)
    print(landmarks_sample.shape)
    print("Loading images...")
    images = np.array([np.array(Image.open(image_folder + fname).convert('L')) / 255 for fname in filenames_sample])

    #convert from dataframe to np
    landmarks_sample = landmarks_sample.to_numpy().astype('float32') 

    print("UTK shape.")
    print(images.shape)
    print(landmarks_sample.shape)

    return images , landmarks_sample

def get_kpts(file_path):
  kpts = []
  f = open(file_path, 'r')
  ln = f.readline()
  while not ln.startswith('n_points'):
      ln = f.readline()

  num_pts = ln.split(':')[1]
  num_pts = num_pts.strip(' ')
  # checking for the number of keypoints
  if float(num_pts) != 68:
      print("keypoint count error")
      return None

  # skipping the line with '{'
  ln = f.readline()

  ln = f.readline()
  while not ln.startswith('}'):
      vals = ln.split(' ')[:2]
      vals = list(map(str.strip, vals))
      vals = list(map(np.float32, vals))
      kpts.append(vals[0])
      kpts.append(vals[1])
      ln = f.readline()

  kpts = np.array(kpts)
  kpts = kpts[72:96] #get eye data only
  return kpts

def load300WSet():
  raw_data_dir = dir_300w_dataset

  files_lfpw = os.listdir(raw_data_dir)
  files_lfpw = [i for i in files_lfpw if i.endswith('.pts')]

  images = []
  points = []
  for index, file_pts in enumerate(files_lfpw):
      file_path = "%s/%s" %(raw_data_dir, file_pts)
      kpts =  get_kpts(file_path)
      
      if kpts is None:
          continue
      
      points.append(kpts)

      file_jpg = file_pts.split('.')[0] + '.jpg'
      jpg_path =  "%s/%s" %(raw_data_dir, file_jpg)
      if not os.path.isfile(jpg_path):
          file_jpg = file_pts.split('.')[0] + '.png'
          jpg_path =  "%s/%s" %(raw_data_dir, file_jpg)

      images.append(np.array(Image.open(jpg_path).convert('L')) / 255)

  #The 300W dataset is made up of images of different sizes. Here we transform them to be
  #200,200 as expected by the model and to match the UTK dataset
  transform = A.Compose([
    A.LongestMaxSize(max_size=200, interpolation=cv2.INTER_LINEAR),
    A.PadIfNeeded(min_height=200, min_width=200, border_mode=cv2.BORDER_CONSTANT, value=0),
  ],keypoint_params=A.KeypointParams(format='xy'))

  image_numpy, points_numpy = augment_images(images, points, transform, 24)

  return image_numpy , points_numpy

        
print("Loading data...")
x_set1, y_set1 = loadUTKSet()
x_set2, y_set2 = load300WSet()

new_images = np.concatenate([x_set1, x_set2])
new_landmarks = np.concatenate([y_set1, y_set2])

print("Splitting data...")
x_train, x_val, y_train, y_val = train_test_split( new_images , new_landmarks , test_size=0.2 )

print(x_val.shape)
print(y_val.shape)

test_size = 5
x_test = x_val[-test_size:]
y_test = y_val[-test_size:]

print(x_test.shape)
print(y_test.shape)

x_val = x_val[:-test_size]
y_val = y_val[:-test_size]

print(x_val.shape)
print(y_val.shape)

batch_size = 32
my_training_batch_generator = My_Custom_Generator(x_train, y_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(x_val, y_val, batch_size)

inputs = tf.keras.layers.Input(shape=(200, 200, 1))
x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7))(inputs)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Conv2D(128, kernel_size=(5, 5))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(128, kernel_size=(5, 5))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Conv2D(256, kernel_size=(5, 5))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(256, kernel_size=(5, 5))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(512, kernel_size=(5, 5))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=4096, kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dense(units=4096)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dense(units=24)(x)

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96
)

model = tf.keras.Model(inputs, outputs=x)
model.compile( loss=tf.keras.losses.MeanSquaredError() , optimizer=tf.keras.optimizers.Adam( learning_rate=learning_rate_schedule ) , metrics=[ 'mse' ] )
model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorlog/", write_graph=True, write_images=True)
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
trainingStopCallback = stopOnLoss()

model.fit( x=my_training_batch_generator ,
        validation_data = my_validation_batch_generator,
        epochs=500,
   callbacks=[tensorboard_callback, earlyStopping_callback, trainingStopCallback] )

timestr = time.strftime("%Y%m%d-%H%M%S")
model.save(dir_model_save_folder+'/model_' + timestr + '.keras')

fig = plt.figure(figsize=( 50 , 50 ))

for i in range( 1 , 6 ):
    sample_image = np.reshape( x_val[i] * 255  , ( 200 , 200 ) ).astype( np.uint8 )
    pred = model.predict( x_val[ i : i +1  ] )
    pred = pred.astype( np.int32 )
    print(pred)
    print(pred.shape)
    xy_points = np.reshape(pred, (-1, 2))
    fig.add_subplot( 1 , 10 , i )
    plt.imshow( sample_image , cmap='gray' )
    plt.show()
    plt.plot( xy_points[ : , 0 ] , xy_points[ : , 1 ] , 'ro' )
    plt.show()
    plt.savefig(dir_results_output_folder+"/val" + str(i) + ".png")
    
plt.show()
plt.savefig(dir_results_output_folder+"/result_val.png")

plt.clf()
fig = plt.figure(figsize=( 50 , 50 ))

for i in range( 0 , test_size ):
    sample_image = np.reshape( x_test[i] * 255  , ( 200 , 200 ) ).astype( np.uint8 )
    image_batch = np.expand_dims(x_test[i], axis=0)
    image_tensor = tf.convert_to_tensor(image_batch)

    pred = model.predict( image_tensor )
    pred = pred.astype( np.int32 )
    print(pred)
    print(pred.shape)
    xy_points = np.reshape(pred, (-1, 2))
    fig.add_subplot( 1 , 10 , i+1 )
    plt.imshow( sample_image , cmap='gray' )
    plt.show()
    plt.plot( xy_points[ : , 0 ] , xy_points[ : , 1 ] , 'ro' )
    plt.show()
    plt.savefig(dir_results_output_folder+"/test" + str(i) + ".png")

plt.show()
plt.savefig(dir_results_output_folder+"/result_test.png")