LATEX (“Lamport’s TEX”) is a typesetting system used to render mathematical expressions. It renders specific sequences of plaintext: for example, the phrase “x = {-b \pm \sqrt{b^2-4ac} \over 2a}.” will be rendered as the quadratic equation. This allows for both full LATEX documents, as well as embedding within other files, such as PDFs and HTML. As such, it is one of the most widely used tools for rendering mathematics and science-related special characters, and sees significant use within academic circles. 
  
Although LATEX syntax is quite intuitive, the staggering number of characters supported by the language and its libraries makes it difficult for beginners. For example, the /prod, /Pi, and /pi characters are rather similar in both their syntax and appearance, but have very different use cases. Memorizing each and every one of LATEX‘s thousands of commands, including many which are so similar, is a daunting task. 
  
To address this issue, I have created a convolutional neural network trained on 130000 labeled greyscale examples of 283 different LaTeX characters obtained from the [Detexify Project] (https://github.com/kirel/detexify-data). They have been cropped, scaled down to 28 by 28, and fed into a CNN with 3 convolutional layers, 2 pooling layers, and 2 fully connected layers. The model was trained for 10 epochs with a batch size of 64. The model achieved a validation and testing accuracy of around 78%. However, as LaTeX contains many characters which are visually similar but syntactically different (\prod and \Pi being some examples), the model is meant to suggest a set of possible commands which appear similar to the input. When evaluated on whether the correct command was in its top 3 suggestions, the model achieved an accuracy of 94.6% on the testing set.

## Usage

main.py contains most of the essential code. At the bottom of the file, it contains several functions for testing out the model, along with descriptions. By default, it will prompt the user to draw a character, then give its best predictions. Tensorboard logs are cached in the ./tensorboard directory, and can be viewed with the function at the bottom of main.py.

This project requires the following libraries:
- Python 3.9.5
- Tensorflow 2.13.0
- Numpy 1.24.3
- OpenCV ("cv2") 4.8.0

The project was successfully run on MacOS Ventura 13.5 using a MacBook Air with Intel i5 cpu, but should work on any OS.
