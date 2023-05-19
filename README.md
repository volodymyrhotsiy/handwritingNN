
-Volodymyr Hotsiy 

The model showed 0.9076 accuracy after training and 0.8327 on testing_set


1)-model.py:

In this model.py file, I have implemented a neural network for training and testing on the 
EMNIST ByClass dataset, which contains uppercase letters, lowercase letters, and digits(62). 
The first step is to import the necessary libraries, including TensorFlow and TensorFlow Datasets.

I then define some constants such as AUTOTUNE and BATCHSIZE. AUTOTUNE allows 
TensorFlow to automatically optimize the performance of data loading, while BATCHSIZE
determines the number of samples in each batch during training.

To ensure that the input images are in a suitable range for neural network training, I define a 
normalize_image function that normalizes the pixel values to the range [0, 1].

Next, I load the EMNIST dataset, splitting it into training and testing subsets. I also specify 
that the data should be loaded in a shuffled manner and with additional information about the dataset.

For the training set, I apply the normalize_image function to each image-label pair, using 
parallel processing for efficiency. I then cache the preprocessed data, shuffle it, create 
batches of size BATCHSIZE, and prefetch it to improve performance.

Similarly, for the testing set, I apply normalization, create batches with a fixed size of 128, and 
prefetch the data.

Moving on to the neural network architecture, I define a sequential model using the Keras 
API. The model consists of an input layer that expects input images with dimensions (28, 28,
1), followed by a convolutional layer with 62 filters and a kernel size of 3, using the ReLU 
activation function. The output of the convolutional layer is flattened, and then passed
through a dense layer with 62 units.

I compile the model by specifying the optimizer, loss function, and evaluation metric. In this case, 
I use the Adam optimizer with a learning rate of 0.001, the sparse categorical cross-
entropy loss function, and accuracy as the metric.

To train the neural network, I call the fit method on the model, passing in the 
preprocessed training data (ds_train). I train the model for 10 epochs and set the 
verbosity level to 2.

After training, I evaluate the performance of the neural network on the testing data 
(ds_test) using the evaluate method. This provides metrics such as loss and accuracy.

Finally, I check if a command-line argument is provided for the filename to save the model. If 
so, I save the model using the specified filename and print a message indicating the 
successful save operation.

Notes: this code takes the name of our stored model as CL argument so you run it like this - python model.py *model_name*


2)-test_infer.py:

In this code snippet, I have written a script that performs image classification inference. 
The script expects a directory path as a command-line argument, which should point to the directory 
containing the input images for classification.

First, I import the necessary libraries, including sys, argparse, and tensorflow. 
These libraries enable me to handle command-line arguments, perform file operations, and 
utilize the TensorFlow framework for image classification.

Next, I define a dictionary called label_to_ascii, which maps the predicted label indices to 
their corresponding ASCII values. This will be useful for printing the results later.

I create an argument parser using the argparse library to handle command-line arguments. 
The script requires the --input argument, which specifies the path to the input directory containing the images to classify.

After parsing the command-line arguments, I check if the specified input directory exists. 
If it doesn't, I print an error message and exit the script.

Next, I retrieve a list of image files in the specified input directory. This is done by iterating 
over the files in the directory and searching for files with extensions .png, .jpg, or .jpeg. I add the file paths to the image_files list.

I load the trained model using TensorFlow's load_model function.

I then iterate over each image file in the image_files list and perform inference on each image. 
For each image, I load and preprocess it using TensorFlow's image preprocessing functions. 
The image is resized to (28, 28) and converted to grayscale.

I pass the preprocessed image to the model's predict function to obtain the predicted label probabilities. 
I use argmax to determine the predicted label index and convert it to a numpy array for easier access.

Finally, I print the predicted ASCII value corresponding to the label index, along with the image file path. 
This allows me to see the predicted result for each image in the directory.


Usage instruction:

Run python test_infer.py --input *your dir name*