from os import path
from pydub import AudioSegment
import librosa
import np

def mp3towav(srcMP3File, destWavFile):

    # convert wav to mp3
    sound = AudioSegment.from_mp3(srcMP3File)
    sound.export(destWavFile, format="wav")

    #https://pythonbasics.org/convert-mp3-to-wav/



#saveSound
#input - sound as STFT data array
#output - WAV file
def saveSound(sound, sr, wavFile):
    out = librosa.istft(sound)
    librosa.output.write_wav('file_trim_5s.wav', out, sr)
    pass

#wavFileToSound
#input - wav file name of path from ALD/
#output- STFT shifted data array
def wavFileToSound(wavFile):
    y, sr = librosa.load(wavFile)
    shiftedData = np.abs(librosa.stft(y))
    return shiftedData


def playWavFile(wavFile):
    #TODO
    pass

