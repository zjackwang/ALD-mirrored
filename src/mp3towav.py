


def mp3towav(srcMP3File, destWavFile):
    from os import path
    from pydub import AudioSegment

    # files
    src = "transcript.mp3"
    dst = "test.wav"

    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

    #https://pythonbasics.org/convert-mp3-to-wav/

def saveSound(sound, wavFile):
    #TODO
    pass

def wavFileToSound(wavFile):
    #TODO
    pass
def playWavFile(wavFile):
    #TODO
    pass



