
from src.mp3towav import *

def playWavFile(wavFile):
    import os
    os.system("aplay "+wavFile)




wavFile = "smallsounds/SlideClean.wav"
sound = wavFileToSound(wavFile)
print(np.sum(sound))

print(sound)