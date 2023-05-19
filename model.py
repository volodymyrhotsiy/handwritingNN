import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras 

# Define constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCHSIZE = 64

# Normalizing image
def normalize_image(image,label):
    return tf.cast(image, tf.float32)/255.0, label

# Prepare data for training
(ds_train, ds_test), ds_info = tfds.load('emnist',split=['train', 'test'],
                                         shuffle_files=True
                                         ,as_supervised=True,
                                         with_info=True)
# Training set
ds_train = ds_train.map(normalize_image,num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCHSIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

#Testing set
ds_test = ds_test.map(normalize_image,num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)

# Create a convolutional neural network
model = tf.keras.models.Sequential([
    keras.Input((28, 28, 1)),
    keras.layers.Conv2D(62, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(62), 
])

# Train neural network
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(ds_train, epochs=10, verbose=2)

# Evaluate neural network performance
model.evaluate(ds_test)

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")