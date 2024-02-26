# Audio Recognition Model

This project demonstrates how to preprocess audio files in the MP3 format, convert them to WAV and build and train a automatic speech recognition (ASR) model for recognizing ten different words. I used a small dataset, which contains short audio clips of commands, such as "اوراق","ارز","سکه","بانک","طلا","نفت","مشتقات","فلزات","صندوق سهامی","صندوق درآمد ثابت","صندوق مختلط","صندوق قابل معامله".

This project has several parts:

## Preprocessing
Did these methods to my dataset of MP3 files with 44KHz sample rate:
- Converting MP3 fils to WAV
- Downsampling From 44KHz to 16KHz
- Renaming each sub-directories to its labels


## Data Augmentation
I produced more data from original dataset using these methods:
- Pitch Shifting
- Time Shifting
- Time Stretching


## Feature Extraction
In this project, I employed wav to spectrograms to extract features from spectrograms.



## Model Shape
Following the splitting of the dataset into training and testing sets, I proceed to standardize them and used this structure for model :

- The Resizing layer resizes the input spectrograms to a target size of 70x64 pixels.
- The Normalization layer normalizes the input spectrograms.
- Two Conv2D layers with 32 and 16 filters, respectively, apply convolutional operations to extract features from the input data. The activation function used is ReLU.
- The MaxPooling2D layer performs down-sampling, reducing the spatial dimensions of the feature maps while preserving important information.
- The Flatten layer flattens the multi-dimensional feature maps into a 1D vector.
- The Dense layer with 64 units and ReLU activation applies a fully connected layer to learn higher-level representations.
- The Dropout layer helps prevent overfitting by randomly dropping out a fraction of the connections during training.
- The final Dense layer with num_labels units and softmax activation produces the output probabilities for each class.
