import tensorflow as tf
import numpy as np
from data.labels import index_to_command
import os
import tkinter as tk
from data_formatting import *
import threading as th
import webbrowser
import time
import matplotlib.pyplot as plt

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
BATCH_SIZE = 64


def load_dataset():
    # Loads, shuffles, and splits the processed dataset
    dataset = tf.data.Dataset.load("./data/TF_dataset_all_index").shuffle(300000, seed=1234,
                                                                          reshuffle_each_iteration=False).batch(
        BATCH_SIZE)
    dataset_size = len(dataset)
    train = dataset.take(round(dataset_size * TRAIN_SPLIT))
    test = dataset.skip(round(dataset_size * TRAIN_SPLIT))
    val = test.skip(round(dataset_size * VAL_SPLIT))
    test = test.take(round(dataset_size * VAL_SPLIT))
    return train, val, test


def train_model():
    # Builds, trains, and tests a CNN on the dataset
    # Saves checkpoints and tensorboard logs, as well as the completed model
    train, validation, test = load_dataset()

    example = train.take(1).get_single_element()
    image_shape = example[0][0].shape
    labels_shape = len(index_to_command)

    callbacks = []

    logdir = "./tensorboard/"
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1))

    checkpoint_dir = "./checkpoints/"
    for file in os.listdir(checkpoint_dir):
        os.remove(checkpoint_dir + "/" + file)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/model",
                                                        save_weights_only=True, verbose=0,
                                                        save_best_only=True, monitor="val_loss", mode="min"))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), input_shape=image_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2D(128, (3, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(labels_shape, activation="softmax"))
    # sparse_categorical_crossentropy used because labels are integers; handles equivalent of one-hot encoding and
    # saves on disk space
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train, batch_size=BATCH_SIZE, epochs=10, validation_data=validation, callbacks=callbacks)
    model.load_weights(checkpoint_dir + "/model")
    model.save("./saves/model")
#    model.save("./saves/model_prototype")
    model.summary()
    model.evaluate(test)
    plot_history(history)


def plot_history(history):
    # Plots the model's accuracy and loss over time
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim([0, 1])
    plt.legend(['validation', 'train'], loc='upper right')
    plt.savefig("./plots/accuracy.png")
    plt.show()
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['loss'], label='loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Sparse Categorical Crossentropy)')
    plt.legend(['validation', 'train'], loc='upper right')
    plt.savefig("./plots/loss.png")
    plt.show()


def show_plots():
    # Shows the plots of the model's accuracy and loss over time
    cv2.imshow('Accuracy (press "0" to continue)', cv2.imread("./plots/accuracy.png"))
    cv2.waitKey(0)
    cv2.imshow('Loss (press "0" to continue)', cv2.imread("./plots/loss.png"))
    cv2.waitKey(0)


def load_model():
    # Loads the saved model
    return tf.keras.models.load_model("./saves/model")


def get_command(model, image):
    # Returns the model's top prediction for the given image
    return index_to_command[tf.argmax(model.predict(tf.expand_dims(tf.constant(image), 0), verbose=0), 1).numpy()[0]]


def evaluate_top_three_accuracy(model):
    # For each example in the test set, checks whether the correct label is in the model's top three predictions
    _, _, test = load_dataset()
    correct = 0
    total = 0
    for batch in test:
        prediction = model.predict(batch[0], verbose=0)
        result = tf.cast(tf.math.in_top_k(batch[1], prediction, 3), tf.uint8)
        correct += int(tf.reduce_sum(result))
        total += len(result)
    print(f"Top 3 accuracy: {correct / total}")
    return correct / total


def get_top_three(model, example):
    # Returns the model's top three predictions for the given image
    prediction = model.predict(tf.expand_dims(tf.constant(example), 0), verbose=0)
    result = tf.math.top_k(prediction, 3)
    return [index_to_command[list(result.indices[0].numpy())[i]] for i in range(3)]


def prompt_for_draw():
    # Prompts the user to draw a symbol and returns the resulting bitmap, scaled to a format the model can use
    root = tk.Tk()
    root.title("Draw a LaTeX symbol here! (Press Enter when done)")
    root.geometry("500x500")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=500, height=500, bg="white")
    canvas.pack()
    canvas.pen_down = False
    root.bind("<KeyPress-Return>", lambda event: end(event))
    canvas.bind("<B1-Motion>", lambda event: draw(event, canvas))
    canvas.bind("<Button-1>", lambda event: pen_down(event, canvas))
    canvas.bind("<ButtonRelease-1>", lambda event: pen_up(event, canvas))
    canvas.bitmap = np.zeros((500, 500))

    def pen_down(event, canvas):
        canvas.pen_down = True
        canvas.last_x = event.x
        canvas.last_y = event.y

    def pen_up(event, canvas):
        canvas.pen_down = False

    def draw(event, canvas):
        if canvas.pen_down:
            canvas.create_line(canvas.last_x, canvas.last_y, event.x, event.y, fill="black", width=1)
            for point in get_line((canvas.last_x, canvas.last_y), (event.x, event.y)):
                canvas.bitmap[point] = 1.0
            canvas.last_x = event.x
            canvas.last_y = event.y

    def end(event):
        canvas.destroy()
        root.destroy()

    root.mainloop()

    if not canvas.bitmap.any():
        return None

    bitmap = tf.constant(canvas.bitmap)

    max_x = tf.where(bitmap)[:, 0].numpy().max()
    min_x = tf.where(bitmap)[:, 0].numpy().min()
    max_y = tf.where(bitmap)[:, 1].numpy().max()
    min_y = tf.where(bitmap)[:, 1].numpy().min()

    bitmap = tf.expand_dims(bitmap, -1)
    if max_x - min_x > max_y - min_y:
        bitmap = tf.image.crop_to_bounding_box(bitmap, min_x, min_y, max_x - min_x + 1, max_x - min_x + 1)
    else:
        bitmap = tf.image.crop_to_bounding_box(bitmap, min_x, min_y, max_y - min_y + 1, max_y - min_y + 1)

    return tf.transpose(tf.math.ceil(tf.image.resize(bitmap, (28, 28), method="area")), perm=[1, 0, 2])


def load_tensorboard():
    # Loads tensorboard in the browser to view metrics and graph of the model
    def loadlocalhost():
        time.sleep(5)
        webbrowser.open("localhost:6006")

    th.Thread(target=lambda: loadlocalhost).start()
    try:
        print("Loading tensorboard...")
        os.system("tensorboard --logdir ./tensorboard/")
    except OSError:
        print("Couldn't activate tensorboard. Make sure tensorboard is installed ('pip3 install tensorboard'), "
              "run 'tensorboard --logdir ./tensorboard/', then visit localhost://6006 in your browser")


def show_examples():
    # Shows two examples from the dataset
    data = tf.data.Dataset.load("./data/TF_dataset_all_index")
    ex1 = data.take(1).get_single_element()
    ex2 = data.skip(700).take(1).get_single_element()
    show(ex1[0])
    print(ex1[1].numpy(), f"({index_to_command[ex1[1].numpy()]})")
    show(ex2[0])
    print(ex2[1].numpy(), f"({index_to_command[ex2[1].numpy()]})")


if __name__ == "__main__":
    #    Creates a Tensorflow dataset from the raw sql file
    #    no need to run a server, it just reads the file
    # sql_to_dataset()

    #    Trains the model and saves it along with checkpoints and tensorboard
    # train_model()

    #    Loads the trained model from the save file
    model = load_model()

    #    Evaluates the model's top three accuracy on the test set. Since LaTeX has many characters that look nearly
    #    identical but have syntactic differences, this model was intended to give the user a few options to choose
    #    from. Therefore, the top three accuracy gives a better idea of how well the model performs in practice.
    # evaluate_top_three_accuracy(model)

    #    Prompts the user to draw a symbol and returns the model's top prediction
    # print(get_command(model, prompt_for_draw()))

    #    Prompts the user to draw a symbol and returns the model's top three predictions in order
    print("Top three predictions:")
    for cmd in get_top_three(model, prompt_for_draw()):
        print(cmd)

    #    Loads tensorboard in the browser to view metrics and graph of the model
    # load_tensorboard()

    #    Shows the plots of the model's accuracy and loss over time (plots were made using matplotlib, and are a
    #    little more detailed than Tensorboard)
    show_plots()

    #    Shows two examples from the dataset
    # show_examples()
