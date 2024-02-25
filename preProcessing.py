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
import librosa
import IPython as ipd
from scipy.io import wavfile
from scipy.signal import resample
import soundfile as sf
import tensorflow_io as tfio
DATA_PATH = "data"

def timeStretching():
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                wav = os.path.join(root, file)
                data, sample_rate = sf.read(wav)
                factor = 0.8
                wav_time_stch = librosa.effects.time_stretch(data,rate = factor)
                out = os.path.join(root, 'time_stretching'+ file)
                print(f'Time stretching of the {wav}')
                sf.write(out, wav_time_stch, sample_rate)


def lowPitchShifting():
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                wav = os.path.join(root, file)
                data, sample_rate = sf.read(wav)
                wav_pitch_sf = librosa.effects.pitch_shift(data,sr = sample_rate,n_steps=-2)
                out = os.path.join(root, 'low_pitch_shift'+ file)
                print(f'change pitch the wave {wav}')
                sf.write(out, wav_pitch_sf, sample_rate)

def highPitchShifting():
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                wav = os.path.join(root, file)
                data, sample_rate = sf.read(wav)
                wav_pitch_sf = librosa.effects.pitch_shift(data,sr = sample_rate,n_steps=2)
                out = os.path.join(root, 'high_pitch_shift'+ file)
                print(f'change pitch the wave {wav}')
                sf.write(out, wav_pitch_sf, sample_rate)


def timeShifting():
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                wav = os.path.join(root, file)
                data, sample_rate = sf.read(wav)
                wav_roll = np.roll(data,int(sample_rate/10))
                out = os.path.join(root, 'time_shift'+ file)
                print(f'Shfiting {wav} by Times {sample_rate/10}')
                sf.write(out, wav_roll, sample_rate)

def renameDirs():
       
    path = DATA_PATH 
    folder_names = readDataLabels("dataLabels.json")
    folders = sorted(os.listdir(path), key= int)
    for i, folder in enumerate(folders):
        print(i)
        print(f"folder name : {folder}")
        old_folder_path = os.path.join(path, folder)
        new_folder_name = folder_names[i]
        new_folder_path = os.path.join(path, new_folder_name)
        os.rename(old_folder_path, new_folder_path)

    print("Folders have been renamed.")

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
        data = np.array(json_data)
        return data

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

            sound.export(wav_path, format="wav")
            os.remove(mp3_file)
            print(f"Converted '{mp3_file}' to '{wav_path}' successfully!")
        except Exception as e:
            print(f"Error converting '{mp3_file}': {e}")

def getSpectrogram(waveform,nfft=216,window=512,stride = 256):
    spectrogram = tfio.audio.spectrogram(
    waveform, nfft=nfft, window=window, stride=stride)
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

def makeSpecDs(ds,train=False):
  if train == False:
    return ds.map(
        map_func=lambda audio,label: (getSpectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)
  else:
      temp = ds.map(
        map_func=lambda audio,label: (getSpectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)
      for i in range(20):
            temp = tf.data.Dataset.concatenate(temp, ds.map(
            map_func=lambda audio,label: (getSpectrogram(audio,128+(i*32),128+(i*32),64+(i*32)), label),
            num_parallel_calls=tf.data.AUTOTUNE))
      return temp
