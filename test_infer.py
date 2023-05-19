import os
import sys
import argparse
import tensorflow as tf

# Map labels to coresponding ascii values
label_to_ascii = {
    0: 65,  # A
    1: 66,  # B
    2: 67,  # C
    3: 68,  # D
    4: 69,  # E
    5: 70,  # F
    6: 71,  # G
    7: 72,  # H
    8: 73,  # I
    9: 74,  # J
    10: 75,  # K
    11: 76,  # L
    12: 77,  # M
    13: 78,  # N
    14: 79,  # O
    15: 80,  # P
    16: 81,  # Q
    17: 82,  # R
    18: 83,  # S
    19: 84,  # T
    20: 85,  # U
    21: 86,  # V
    22: 87,  # W
    23: 88,  # X
    24: 89,  # Y
    25: 90,  # Z
    26: 97,  # a
    27: 98,  # b
    28: 99,  # c
    29: 100,  # d
    30: 101,  # e
    31: 102,  # f
    32: 103,  # g
    33: 104,  # h
    34: 105,  # i
    35: 106,  # j
    36: 107,  # k
    37: 108,  # l
    38: 109,  # m
    39: 110,  # n
    40: 111,  # o
    41: 112,  # p
    42: 113,  # q
    43: 114,  # r
    44: 115,  # s
    45: 116,  # t
    46: 117,  # u
    47: 118,  # v
    48: 119,  # w
    49: 120,  # x
    50: 121,  # y
    51: 122,  # z
    52: 48,  # 0
    53: 49,  # 1
    54: 50,  # 2
    55: 51,  # 3
    56: 52,  # 4
    57: 53,  # 5
    58: 54,  # 6
    59: 55,  # 7
    60: 56,  # 8
    61: 57  # 9
}

# Create an argument parser
parser = argparse.ArgumentParser(description="Image classification inference script")
parser.add_argument("--input", required=True, help="Path to the input directory")

# Parse the command-line arguments
args = parser.parse_args()

# Get the input directory path
directory = args.input

# Check if the directory exists
if not os.path.isdir(directory):
    print(f"The directory '{directory}' does not exist.")
    sys.exit(1)

# Get the list of image files in the directory
image_files = [
    os.path.join(directory, file)
    for file in os.listdir(directory)
    if file.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Load the trained model
model = tf.keras.models.load_model("model")

# Iterate over the image files and perform inference
for image_file in image_files:
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(28, 28), color_mode="grayscale")
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)

    # Perform prediction 
    prediction = model.predict(image, verbose=0)
    predicted_value = tf.argmax(prediction, axis=1).numpy()[0]  

    # Print the result 
    print(f"{label_to_ascii[predicted_value]},{image_file}")
