from whisper import load_model
from whisper import load_audio
from whisper import pad_or_trim
from whisper import log_mel_spectrogram
from whisper import DecodingOptions
from whisper import decode

model = load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = load_audio("C://Users/fujitsu/Desktop/Georgia/CS7643_DeepLearning/whisper/audio.mp3")

audio = pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = DecodingOptions(fp16=False)
result = decode(model, mel, options)

# print the recognized text
print(result.text)