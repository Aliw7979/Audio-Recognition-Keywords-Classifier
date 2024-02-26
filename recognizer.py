import tensorflow as tf
import soundfile as sf
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
        if not address.endswith(".wav"):
            raise InvalidFileFormatError("Audio file format should be WAV.")
        else:
            data, sample_rate = sf.read(address)
            if sample_rate != REQUIRED_SAMPLE_RATE:
                print("Warning : Target file should be in Wav format")
                try:
                    prediction = self._loaded_model(address)
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
recognizer.predict("1-(3).wav")