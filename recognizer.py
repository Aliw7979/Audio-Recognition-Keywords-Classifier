import tensorflow as tf
import librosa
import soundfile as sf
import os
REQUIRED_SAMPLE_RATE = 16000
class InvalidFileFormatError(Exception):
    pass

class Recognizer:
    _loaded_model = None
    def __init__(self, address):
        self.address = address

    def load_model(self):
        try:
            self._loaded_model = tf.saved_model.load(self.address)
        except Exception as e:
            print(e)

    def predict(self, address : str):
      try:
        if address.endswith(".mp3"):
            audio, sample_rate = librosa.load(address, sr=None)
            wav_file = address + ".wav"
            sf.write(wav_file, audio, sample_rate, format='wav')
            os.remove(wav_file)
            audio, sample_rate = librosa.load(address, sr=None)
            if sample_rate != REQUIRED_SAMPLE_RATE:
                try:
                    if sample_rate != REQUIRED_SAMPLE_RATE:
                        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr = REQUIRED_SAMPLE_RATE)
                        print("changed sample rate of file.")   
                    
                    desired_length = REQUIRED_SAMPLE_RATE * 4
                    if len(audio) < desired_length:
                        audio = tf.pad(audio, [[0, desired_length - len(audio)]])
                    else:
                        audio = audio[:desired_length]

                    audio = tf.expand_dims(audio, axis=0)
                    prediction = self._loaded_model(audio)
                    result = prediction['class_names'][0].numpy().decode('utf-8')
                    print(f"Model Predicted audio as : {result}")
                except Exception as e:
                    print(e)
            
        elif address.endswith('.wav'):
            audio, sample_rate = librosa.load(address, sr=None)
            if sample_rate != REQUIRED_SAMPLE_RATE:
                print("Warning : Sample rate is not suitable for the model.")
                try:
                    if sample_rate != REQUIRED_SAMPLE_RATE:
                        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr = REQUIRED_SAMPLE_RATE)
                        print("changed sample rate of file.")   
                    
                    desired_length = REQUIRED_SAMPLE_RATE * 4
                    if len(audio) < desired_length:
                        audio = tf.pad(audio, [[0, desired_length - len(audio)]])
                    else:
                        audio = audio[:desired_length]

                    audio = tf.expand_dims(audio, axis=0)
                    prediction = self._loaded_model(audio)
                    result = prediction['class_names'][0].numpy().decode('utf-8')
                    print(f"Model Predicted audio as : {result}")
                except Exception as e:
                    print(e)
            else:
                try:
                    prediction = self._loaded_model(address)
                    result = prediction['class_names'][0].numpy().decode('utf-8')
                    print(f"Model Predicted audio as : {result}")
                except Exception as e:
                    print(e)
      except Exception as e:
        print(e)

recognizer = Recognizer('model')
recognizer.load_model()
recognizer.predict("1-(1).mp3")