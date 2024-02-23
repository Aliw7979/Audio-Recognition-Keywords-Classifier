from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import keras
import pathlib
import preProcessing
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

print(tf.__version__)
def main():
    UNSORTED_DATA_PATH = 'unsortedData'
    SORTED_DATA_PATH = 'data'
    DATA_LABELS_PATH = 'dataLabels.json'
    
    
    data_dir = pathlib.Path(SORTED_DATA_PATH)
    if not data_dir.exists():
        try:
            preProcessing.sortDirectories(UNSORTED_DATA_PATH)

        except:
            print("directory does not exist!")
    
    preProcessing.convert_mp3_to_wav(SORTED_DATA_PATH)
    preProcessing.downsampleTo16K()
    preProcessing.dataInfo()
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000*6,
    subset='both')
    dataLabels = np.array(train_ds.class_names)
    print(train_ds.element_spec)
    train_ds = train_ds.map(preProcessing.squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(preProcessing.squeeze, tf.data.AUTOTUNE)
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)
    for example_audio, example_labels in train_ds.take(1):
        print(example_audio.shape)
        print(example_labels.shape)

    #show plot of few waveforms 
    plt.figure(figsize=(16, 10))
    rows = 3
    cols = 3
    n = rows * cols
    for i in range(n):
      plt.subplot(rows, cols, i+1)
      audio_signal = example_audio[i]
      plt.plot(audio_signal)
      plt.title(dataLabels[example_labels[i]])
      plt.yticks(np.arange(-1.2, 1.2, 0.2))
      plt.ylim([-1.1, 1.1])
    
    plt.tight_layout()
    plt.savefig('waveformExamples.png')

    #convert to spectrogram
    for i in range(3):
        label = dataLabels[example_labels[i]]
        waveform = example_audio[i]
        spectrogram = preProcessing.getSpectrogram(waveform)
        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Spectrogram shape:', spectrogram.shape)

    #show plot for spectrograms
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 96000])

    preProcessing.plotSpectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.suptitle(label.title())
    plt.tight_layout()
    plt.savefig('spectrogramsExamples.png')
    
    #create spectrogram datasets from the audio datasets
    train_spectrogram_ds = preProcessing.makeSpecDs(train_ds)
    val_spectrogram_ds = preProcessing.makeSpecDs(val_ds)
    test_spectrogram_ds = preProcessing.makeSpecDs(test_ds)

    #Examine spectrograms 
    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break
    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

    for i in range(n):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        preProcessing.plotSpectrogram(example_spectrograms[i].numpy(), ax)
        ax.set_title(dataLabels[example_spect_labels[i].numpy()])

    plt.tight_layout()
    plt.savefig('spectrogramsForEachData.png')
    
    #build the model and training
    #reduce read latency while training the model:
    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    #build
    input_shape = example_spectrograms.shape[1:]
    print('Input shape:', input_shape)
    num_labels = len(dataLabels)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = keras.layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        # Downsample the input.
        keras.layers.Resizing(64, 64),
        # Normalize.
        norm_layer,
        keras.layers.Conv2D(64, 4, activation='tanh'),
        keras.layers.Conv2D(32, 4, activation='tanh'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='tanh'),
        keras.layers.Dropout(0.6),
        keras.layers.Dense(num_labels,activation = 'softmax'),
    ])

    print(model.summary())
    
    #adam optimizer
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)
    EPOCHS = 20
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    #plot the results
    metrics = history.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')
    plt.tight_layout()
    plt.savefig('Loss.png')

    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.tight_layout()
    plt.savefig('Accuracy.png')

    #print model eval
    print(model.evaluate(test_spectrogram_ds, return_dict=True))
    model.save("savedModel", save_format="tf")

if __name__ == "__main__":
    main()
