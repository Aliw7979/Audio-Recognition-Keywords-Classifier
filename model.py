from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import pathlib
import preProcessing
import tensorflow as tf

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
    dataLabels = preProcessing.readDataLabels(DATA_LABELS_PATH)
    preProcessing.downsampleTo16K()
    
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')
    print(train_ds.element_spec)
    train_ds = train_ds.map(preProcessing.squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(preProcessing.squeeze, tf.data.AUTOTUNE)
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)
    for example_audio, example_labels in train_ds.take(1):
        print(example_audio.shape)
        print(example_labels.shape)

if __name__ == "__main__":
    main()
