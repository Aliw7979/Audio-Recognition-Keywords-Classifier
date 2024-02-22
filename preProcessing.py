import os
import glob
from pydub import AudioSegment
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from IPython import display
import json
import numpy as np
import pandas as pd
import wave
import subprocess
from scipy.io import wavfile
from scipy.signal import resample
import soundfile as sf
DATA_PATH = "data"

def downsampleTo16K():
    targetSampleRate = 16000
    # Traverse through all subfolders
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            # Check if the file is a WAV file
            if file.endswith(".wav"):
                # Construct the file path
                wav_file = os.path.join(root, file)

                # Read the original WAV file
                data, sample_rate = sf.read(wav_file)

                # Downsample the audio data
                resampled_data = resample(data, int(len(data) * targetSampleRate / sample_rate))

                # Overwrite the original file with the downsampled data
                sf.write(wav_file, resampled_data, targetSampleRate, 'PCM_16')

                print(f"Downsampled {wav_file} to {targetSampleRate} Hz")

def dataInfo():
    subdirectories = []
    bitrates = []
    lengths = []
    for dirpath, dirnames, filenames in os.walk(DATA_PATH):
        for filename in filenames:
            if filename.endswith('.wav'):
                file_path = os.path.join(dirpath, filename)
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    bitrate = frame_rate * sample_width * 8 * channels
                    duration = frames / frame_rate
                    subdirectories.append(dirpath)
                    bitrates.append(bitrate)
                    lengths.append(duration)

    data = {
    'Subdirectory': subdirectories,
    'Average Bitrate': bitrates,
    'Average Length': lengths
    }
    df = pd.DataFrame(data)
    grouped_df = df.groupby('Subdirectory').mean()
    print(grouped_df)

def readDataLabels(address):
    with open(address, "r") as file:
        json_data = json.load(file)
        print(json_data)
        file.close
        return np.array(json_data)

def sortDirectories(address):
    data_dir = pathlib.Path(address)
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands.astype(int).argsort()]
    print("Commands:", commands)
    target_names = [str(i) for i in range(1, 13)]

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Directory '{DATA_PATH}' created successfully!")
    else:
        print(f"Directory '{DATA_PATH}' already exists.")

    for old_name, new_name in zip(commands, target_names):
        old_path = os.path.join(address, str(old_name))
        new_path = os.path.join(DATA_PATH, str(new_name))
        os.rename(old_path, new_path)
    print("Folders renamed successfully!")
    
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def convert_mp3_to_wav(directory, output_directory=None):

    if not output_directory:
        output_directory = directory

    os.makedirs(output_directory, exist_ok=True) 
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            subdirectory_path = os.path.join(directory, subdir)
            convert_mp3_to_wav(subdirectory_path, output_directory=os.path.join(output_directory, subdir))
    for mp3_file in glob.glob(os.path.join(directory, "*.mp3")):
        wav_filename = os.path.splitext(os.path.basename(mp3_file))[0] + ".wav"
        wav_path = os.path.join(output_directory, wav_filename)

        try:
            sound = AudioSegment.from_mp3(mp3_file)

            sound.export(wav_path, format="wav", codec="pcm_s16le")
            os.remove(mp3_file)
            print(f"Converted '{mp3_file}' to '{wav_path}' successfully!")
        except Exception as e:
            print(f"Error converting '{mp3_file}': {e}")

