"""Pre Processing for audio
files as they are input.
For training data.
"""

from __future__ import print_function
import librosa

# 1. Get the file path to the included audio example
filename = "C:\Users\Reece\Documents\GitHub\ALD\Sound files\Slide Whistle 2 - Sound Effect.wav"

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename)

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
