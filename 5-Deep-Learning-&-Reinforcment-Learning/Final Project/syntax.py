import marimo

__generated_with = "0.2.8"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Project 5 - Deep Learning & Reinforcement Learning

        The MNIST dataset is a widely used benchmark in the field of machine learning and computer vision. It consists of a collection of 28x28 pixel grayscale images of handwritten digits (0-9) along with their corresponding labels. MNIST serves as a standard dataset for training and testing various image classification algorithms, making it an essential resource for researchers and practitioners in the field.

        AlexNet, on the other hand, is a deep convolutional neural network architecture designed for image classification. It gained prominence after winning the ImageNet Large Scale Visual Recognition Challenge in 2012, showcasing its effectiveness in image recognition tasks.

        The Alexnet has eight layers with learnable parameters. The model consists of five layers with a combination of max pooling followed by 3 fully connected layers and they use Relu activation in each of these layers except the output layer. They found out that using the relu as an activation function accelerated the speed of the training process by almost six times. They also used the dropout layers, that prevented their model from overfitting. Further, the model is trained on the Imagenet dataset. The Imagenet dataset has almost 14 million images across a thousand classes.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from tqdm import tqdm
    from skimage import transform
    import cv2
    from tensorflow import keras
    from tensorflow.keras import datasets
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.callbacks import EarlyStopping
    return (
        Activation,
        Conv2D,
        Dense,
        Dropout,
        EarlyStopping,
        Flatten,
        ImageDataGenerator,
        MaxPooling2D,
        Sequential,
        cv2,
        datasets,
        ff,
        go,
        keras,
        make_subplots,
        np,
        pd,
        plt,
        px,
        sns,
        tqdm,
        transform,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Define Helper Function
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Show Image
        """
    )
    return


@app.cell
def _(np, plt):
    def plot_images(X_data, y_data, rows=2, cols=5):
        fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

        indices = np.random.choice(len(X_data), size=rows * cols, replace=False)

        for i in range(rows):
            for j in range(cols):
                index = indices[i * cols + j]
                axes[i, j].imshow(X_data[index])
                axes[i, j].set_title(f'Label: {y_data[index]} \n Index {index}')
                axes[i, j].axis('off')

        plt.show()
    return plot_images,


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Preprocessing Image
        """
    )
    return


@app.cell
def _(cv2, np, tqdm, transform):
    def resize_images(images, img_size):
        resized_images = []

        for i in tqdm(range(0, len(images))):
            img = cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB)
            resized_images.append(transform.resize(img, (img_size, img_size)).astype('float32'))

        return np.array(resized_images, dtype='float32')
    return resize_images,


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Plot Prediction Label
        """
    )
    return


@app.cell
def _(np, plt, prediction_prob):
    def plot_image_with_probabilities(X_test, y_test, index=None):
        if index is None:
            index = np.random.randint(0, len(X_test))

        # Get the image and true label
        test_image = X_test[index]
        true_label = y_test[index]

        # Predict probabilities using the model
        probabilities = prediction_prob[index]
        predicted_label = np.argmax(probabilities)

        # Plot the image along with predicted probabilities
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Plot the image
        ax1.imshow(test_image, cmap='gray')
        ax1.set_title(f'Actual: {true_label}')
        ax1.axis('off')

        # Plot the predicted probabilities
        ax2.bar(range(10), probabilities)
        ax2.set_xticks(range(10))
        ax2.set_title(f'Predicted: {predicted_label} with Probabilities')

        plt.tight_layout()
        plt.show()
    return plot_image_with_probabilities,


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Plot Confusion Matrix
        """
    )
    return


@app.cell
def _(classification_report, plt):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    def plot_confusion_matrix(y_true, y_pred):
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create a ConfusionMatrixDisplay object
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)

        # Plot the confusion matrix
        cm_display.plot(cmap='Blues')

        # Customize the plot
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks([0, 9])  # Set x-axis tick labels
        plt.yticks([0, 9])  # Set y-axis tick labels
        plt.show()

        print(classification_report(y_true, y_pred, digits=3))
    return ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix


@app.cell
def _(mo):
    mo.md(
        r"""
        # Read Data
        """
    )
    return


@app.cell
def _(datasets):
    #load dataset
    n_images = 6000

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    X_train = X_train[:6000]
    X_test = X_test[:6000//2]
    y_train = y_train[:6000]
    y_test = y_test[:6000//2]


    print(X_train.shape, X_test.shape)
    print(y_train.shape)
    return X_test, X_train, n_images, y_test, y_train


@app.cell
def _(X_train, plot_images, y_train):
    plot_images(X_train, y_train)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Preprocess Image
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        It's important to highlight that AlexNet, being a deep neural network architecture, introduced padding as a crucial element to mitigate the drastic reduction in the size of feature maps during convolutional operations. The model takes as input images sized 227x227 pixels with 3 color channels. The incorporation of padding is a strategic design choice that helps preserve spatial information and contributes to the network's overall performance.
        """
    )
    return


@app.cell
def _(X_test, X_train, resize_images, y_test, y_train):
    X_train_proc = resize_images(X_train, 227)
    X_test_proc = resize_images(X_test, 227)

    print(X_train_proc.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    return X_test_proc, X_train_proc


@app.cell
def _(X_train_proc, plot_images, y_train):
    plot_images(X_train_proc, y_train)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Modeling AlexNet
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <img src="5-Deep-Learning-&-Reinforcment-Learning/Final Project/alexnet-summary.webp" alt="AlexNet Summary" width="800"/>

        <img src="5-Deep-Learning-&-Reinforcment-Learning/Final Project/alexnet-summary 2.webp" alt="AlexNet Summary" width="800"/>
        """
    )
    return


@app.cell
def _(Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Sequential):
    class AlexNet(Sequential):
        def __init__(self, input_shape, num_classes):
            super().__init__()

            self.add(Conv2D(96, kernel_size = (11,11), strides = 4,
                            padding = 'valid', activation = 'relu',
                            input_shape=input_shape,
                            kernel_initializer = 'he_normal'))
            self.add(MaxPooling2D(pool_size = (3,3), strides=(2,2),
                                  padding = 'valid'))
            
            self.add(Conv2D(256, kernel_size = (5,5), strides = 1,
                            padding = 'same', activation='relu',
                            kernel_initializer = 'he_normal'))
            self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                                  padding = 'valid'))
            
            self.add(Conv2D(384, kernel_size = (3,3), strides = 1,
                            padding = 'same', activation='relu',
                            kernel_initializer = 'he_normal'))
            
            self.add(Conv2D(384, kernel_size = (3,3), strides = 1,
                            padding = 'same', activation='relu',
                            kernel_initializer = 'he_normal'))
            
            self.add(Conv2D(256, kernel_size = (3,3), strides = 1,
                            padding = 'same', activation='relu',
                            kernel_initializer = 'he_normal'))
            
            self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                                  padding = 'valid'))
            
            self.add(Flatten())
            
            self.add(Dropout(rate=0.5))
            self.add(Dense(4096, activation='relu'))
            self.add(Dropout(rate=0.5))
            self.add(Dense(4096, activation='relu'))

            self.add(Dense(num_classes, activation='softmax'))
    return AlexNet,


@app.cell
def _(AlexNet, EarlyStopping, keras):
    early_stopping = EarlyStopping(min_delta = 0.001,patience = 20,restore_best_weights = True,verbose = 0)
    model = AlexNet((227,227,3), num_classes = 10)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)
    model.summary()
    return early_stopping, model


@app.cell
def _(X_test_proc, X_train_proc, early_stopping, model, y_test, y_train):
    history = model.fit(X_train_proc, y_train, batch_size = 256, epochs = 50,
              callbacks = [early_stopping], validation_data= (X_test_proc, y_test))
    return history,


@app.cell
def _(X_test_proc, model, np):
    prediction_prob = model.predict(X_test_proc)
    prediction_class = np.argmax(prediction_prob,axis=1)
    return prediction_class, prediction_prob


@app.cell
def _(X_test_proc, plot_images, prediction_class):
    plot_images(X_test_proc, prediction_class)
    return


@app.cell
def _(plot_confusion_matrix, prediction_class, y_test):
    plot_confusion_matrix(y_test, prediction_class)
    return


@app.cell
def _(X_test_proc, plot_image_with_probabilities, y_test):
    plot_image_with_probabilities(X_test_proc, y_test)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In conclusion, AlexNet proves to be a powerful and effective deep learning architecture for image classification tasks, showcasing its robust performance on the MNIST dataset. With an impressive accuracy of 98.4% on the test data, AlexNet demonstrates its capability to accurately predict and classify handwritten digits, affirming its significance as a formidable tool in the realm of image recognition and deep learning.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()

