import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from os.path import exists
import matplotlib.pyplot as plt
import tensorflow as tf

keras_model_path = ""
test_image_path = "" #200x200 image
results_output_folder = ""

model = tf.keras.models.load_model(keras_model_path)

image = np.array(Image.open(test_image_path).convert('L')) / 255

#not strictly necessary
image_batch = np.expand_dims(image, axis=0)
image_tensor = tf.convert_to_tensor(image_batch)

pred = model.predict( image_tensor )

pred = pred.astype( np.int32 )
xy_points = np.reshape(pred[0], (-1, 2))
fig, ax = plt.subplots()
plt.imshow( image , cmap='gray' )
plt.show()
plt.plot( xy_points[ : , 0 ] , xy_points[ : , 1 ] , 'ro' )
plt.show()
plt.savefig(results_output_folder+"/result.png")
