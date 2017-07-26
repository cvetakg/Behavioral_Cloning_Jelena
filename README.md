# Behavioral_Cloning_Jelena
# PROJECT REVIEW


Meets Specifications

Congratulations, your project meets all specifications! :star2:

I've enjoyed very much reviewing your project, you can be proud of the hard work you've done!

Comma.ai

Few articles:

NVIDIA’s neural network model and image augmentation
Behavioral Cloning - Make a car drive like yourself
Batch size discussion
Using Augmentation to Mimic Human Driving
Cloning a Car to mimic Human Driving
Behavioral Cloning with David Silver
Drawbacks of Behavioral Cloning George Hotz on Comma.ai

Awesome Deep Learning Papers recommended by Kaggle
Required Files

The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.
Quality of Code

The model provided can be used to successfully operate the simulation.

Excellent!! The model provided works successfully in the simulator. :)

The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

*The code is clearly organized and comments are included where needed.!

    Generators are better for huge loops, as they only "hold" one value at the time.
    Corey Schafer's Video on Generators

Model Architecture and Training Strategy

The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.

    Wonderful,the neural network uses convolution layers with appropriate filter sizes.Layers exist to introduce nonlinearity into the model.!
    Data is properly normalized prior training the model.

Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.

    Great job splitting the training dataset into training and validation subsets!
    Dropout (Regularization technique) layers were not used to reduce overfitting.
    Dropout Regularization in Deep Learning Models With Keras
    You might like to learn the whole idea of Dropout :smiley: It's gives a brief analysis of the technique.

Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

Training data was properly chosen to induce the desired driving behavior in the simulation.
Architecture and Training Documentation

The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

Wonderful, your README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

The architecture of the implemented model provides sufficient details of the characteristics and qualities,such as the type of model used, the number of layers, the size of each layer.

The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.

The training process is properly described in the README, great job!
