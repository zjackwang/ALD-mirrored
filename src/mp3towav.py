from os import path
from pydub import AudioSegment

def mp3towav(srcMP3File, destWavFile):

    # convert wav to mp3
    sound = AudioSegment.from_mp3(srcMP3File)
    sound.export(destWavFile, format="wav")
    
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



mp3towav("sounds/FrenchClean.mp3", "sounds/FrenchClean.wav")
