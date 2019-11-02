"""Pre Processing for audio
files as they are input.
For training data.
"""

#from __future__ import print_function
import librosa.display
import librosa
import np

# 1. Get the file path to the included audio example
data_folder = "C:\\Users\\Reece\\Documents\\GitHub\\ALD\\ALD\\src\\"
filename = data_folder + "50FrenchClean.mp3"

filename = librosa.util.example_audio_file()

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename)

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
D = np.abs(librosa.stft(y))

