import tensorflow as tf
import librosa
import soundfile as sf
import os
REQUIRED_SAMPLE_RATE = 16000
RESULT = "Model Predicted audio as : {}"
MP3_EX = ".mp3"
WAV_EX = ".wav"
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
        if address.endswith(MP3_EX):
            audio, sample_rate = librosa.load(address, sr=None)
            wav_file = address + WAV_EX
            sf.write(wav_file, audio, sample_rate, format='wav')
            os.remove(wav_file)
            audio, sample_rate = librosa.load(address, sr=None)
            if sample_rate != REQUIRED_SAMPLE_RATE:
                    if sample_rate != REQUIRED_SAMPLE_RATE:
                        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr = REQUIRED_SAMPLE_RATE)
                    
                    desired_length = REQUIRED_SAMPLE_RATE * 4
                    if len(audio) < desired_length:
                        audio = tf.pad(audio, [[0, desired_length - len(audio)]])
                    else:
                        audio = audio[:desired_length]

                    audio = tf.expand_dims(audio, axis=0)
                    prediction = self._loaded_model(audio)
                    result = prediction['class_names'][0].numpy().decode('utf-8')
                    print(RESULT.format(result))
            else:
                prediction = self._loaded_model(address)
                result = prediction['class_names'][0].numpy().decode('utf-8')
                print(RESULT.format(result))

        elif address.endswith(WAV_EX):
            audio, sample_rate = librosa.load(address, sr=None)
            if sample_rate != REQUIRED_SAMPLE_RATE:
                if sample_rate != REQUIRED_SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr = REQUIRED_SAMPLE_RATE)
                
                desired_length = REQUIRED_SAMPLE_RATE * 4
                if len(audio) < desired_length:
                    audio = tf.pad(audio, [[0, desired_length - len(audio)]])
                else:
                    audio = audio[:desired_length]

                audio = tf.expand_dims(audio, axis=0)
                prediction = self._loaded_model(audio)
                result = prediction['class_names'][0].numpy().decode('utf-8')
                print(RESULT.format(result))
            else:
                prediction = self._loaded_model(address)
                result = prediction['class_names'][0].numpy().decode('utf-8')
                print(RESULT.format(result))

      except Exception as e:
        print(e)

recognizer = Recognizer('model')
recognizer.load_model()
recognizer.predict("1-(1).mp3")