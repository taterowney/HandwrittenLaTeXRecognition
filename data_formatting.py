import tensorflow as tf
import numpy as np
import math
import json
import threading
import cv2


def sql_to_dataset():
    # Reads the raw SQL file and converts it to 50 different datasets (one for each thread)
    # tried to run it as a database but figured that this would be simpler
    # WARNING: takes a long time to run
    # data begins: line 78
    # data ends: line 210532
    with open("./data/detexify.sql") as f:
        data = f.readlines()[78:210531]
        elems_per_thread = (210531 - 78) // 50
        for thread_num in range(50):
            thread = threading.Thread(target=convert_data, args=(
                data[thread_num * elems_per_thread:(thread_num + 1) * elems_per_thread], thread_num))
            thread.run()
    combine_datasets()
    convert_labels()


def convert_data(data, thread_num):
    # Converts the raw to a dataset
    # data is a list of tuples (strokes, labels)
    # thread_num is the number of the thread
    # returns a dataset
    labels = [data[i].split("\t")[1] for i in range(len(data))]
    strokes = [points_to_bitmap(eval(data[i].split("\t")[2])) for i in range(len(data))]
    dataset = tf.data.Dataset.from_tensor_slices((strokes, labels))
    dataset.save(f"./data/TF_dataset{thread_num}")


def show(bitmap):
    # Shows a black/white image in a window using cv2
    bitmap = tf.image.resize(bitmap, (500, 500), method="nearest").numpy()[:, :, 0]
    cv2.imshow('Bitmap (press "0" to continue)', bitmap.astype(np.uint8) * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def points_to_bitmap(points):
    # Converts a list of lines to a bitmap
    # points is a list of curves, each one composed of a list of points it passes through
    bitmap = np.zeros((1000, 1000), np.float32)
    max_x = 0
    max_y = 0
    min_x = 1000
    min_y = 1000
    for curve in points:
        # if the "curve" is just a dot, don't do all the other stuff
        if len(curve) == 1:
            bitmap[math.ceil(curve[0][0]), math.ceil(curve[0][1])] = 1.0
            max_x = math.ceil(curve[0][0])
            min_x = math.ceil(curve[0][0])
            max_y = math.floor(curve[0][1])
            min_y = math.floor(curve[0][1])
            continue
        prev = curve[0]
        for point in range(1, len(curve)):
            # for each point, check whether it is a maximum/minimum (so we know how to crop the resulting image)
            if prev[0] > max_x:
                max_x = math.ceil(prev[0])
            if prev[0] < min_x:
                min_x = math.ceil(prev[0])
            if prev[1] > max_y:
                max_y = math.floor(prev[1])
            if prev[1] < min_y:
                min_y = math.floor(prev[1])
            # draw a line from the previous point to the current point
            for pixel in get_line(prev[:2], curve[point][:2]):
                bitmap[pixel] = 1.0
            prev = curve[point]
    # add a single channel
    bitmap = tf.expand_dims(bitmap, 2)
    if min_x > max_x or min_y > max_y:
        print("weird")
        print(points)
        return tf.zeros((28, 28, 1))
    # crop the image so that it forms a square and doesn't cut out any of the written pixels
    # content will be anchored to the top left corner
    try:
        if max_x - min_x > max_y - min_y:
            bitmap = tf.image.crop_to_bounding_box(bitmap, min_x, min_y, max_x - min_x + 1, max_x - min_x + 1)
        else:
            bitmap = tf.image.crop_to_bounding_box(bitmap, min_x, min_y, max_y - min_y + 1, max_y - min_y + 1)
    except Exception:
        print("weird")
        print(points)
        return tf.zeros((28, 28, 1))
    # resize the image to 28x28
    bitmap = tf.image.resize(bitmap, (28, 28), method="area")
    # transpose the image so that it is no longer sideways
    return tf.transpose(tf.math.ceil(bitmap), perm=[1, 0, 2])


def get_line(from_pt, to_pt):
    # Bresenham's line algorithm
    # written by GPT-3
    # from_pt and to_pt are tuples of the form (x, y)
    # returns a list of points that the line passes through

    x1, y1 = from_pt
    x2, y2 = to_pt
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx

    if steep:
        x1, y1 = y1, x1  # Swap x and y
        x2, y2 = y2, x2  # Swap x and y

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = y2 - y1
    error = dx // 2

    y_step = 1 if y1 < y2 else -1
    y = y1

    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= abs(dy)

        if error < 0:
            y += y_step
            error += dx

    return points


def convert_labels():
    # Converts the labels from strings to indices, and write the indices and names to a dictionary in a python file
    # so that we can reference them later
    dataset = tf.data.Dataset.load("./data/TF_dataset_all").as_numpy_iterator()
    images = []
    labels = []
    categories = []
    for example in dataset:
        # only include examples that are part of the core LaTeX package
        if example[1].decode().startswith("latex"):
            images.append(example[0])
            labels.append(example[1].decode())
            # add the class name to the list of classes if it isn't already there
            if example[1].decode() not in categories:
                categories.append(example[1].decode())
    images = tf.constant(images)
    # since the original dataset contains the IDs of the symbols, we need to convert them to their corresponding
    # commands found in symbols.json from the detexify dataset
    with open("./data/symbols.json", "r") as f:
        symbols = json.load(f)
    id_to_command = {symbols[i]['id']: symbols[i]['command'] for i in range(len(symbols))}
    id_to_index = {categories[i]: i for i in range(len(categories))}
    labels = tf.stack([tf.constant(id_to_index[label]) for label in labels])
    tf.data.Dataset.from_tensor_slices((images, labels)).save("./data/TF_dataset_all_index")
    # write the dictionary to ./data/labels.py
    with open("./data/labels.py", "w") as f:
        f.write("index_to_command = {" + ", ".join(
            [fr"{i}: r'{id_to_command[categories[i]]}'" for i in range(len(categories))]) + "}" + "\n")


def combine_datasets():
    # Combines the different datasets produced on each thread into a single dataset
    dataset = tf.data.Dataset.load("./data/TF_dataset0")
    for i in range(1, 50):
        dataset = dataset.concatenate(tf.data.Dataset.load(f"./data/TF_dataset{i}"))
    dataset.save("./data/TF_dataset_all")


if __name__ == "__main__":
    sql_to_dataset()
