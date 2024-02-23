import os
import glob
from pydub import AudioSegment
import pathlib
import numpy as np
import seaborn as sns
import tensorflow as tf
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
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                wav_file = os.path.join(root, file)
                data, sample_rate = sf.read(wav_file)
                if sample_rate == targetSampleRate:
                    continue
                resampled_data = resample(data, int(len(data) * targetSampleRate / sample_rate))
                sf.write(wav_file, resampled_data, targetSampleRate, 'PCM_16')
                print(f"Downsampled {wav_file} to {targetSampleRate} Hz")

def dataInfo():
    subdirectories = []
    bitrates = []
    lengths = []
    longest = []
    for dirpath, dirnames, filenames in os.walk(DATA_PATH):
        longest_length = 0
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
                    if longest_length < duration:
                        longest_length = duration
        if longest_length != 0:
            longest.append(longest_length)

    data = {
    'Subdirectory': subdirectories,
    'Average Bitrate': bitrates,
    'Average Length': lengths
    }
    df = pd.DataFrame(data)
    grouped_df = df.groupby('Subdirectory').mean()
    grouped_df["longest"] = longest
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

def getSpectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plotSpectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def makeSpecDs(ds):
  return ds.map(
      map_func=lambda audio,label: (getSpectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)