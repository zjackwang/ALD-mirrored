def playWavFile(wavFile):
    import os
    os.system("aplay "+wavFile)




playWavFile("smallsounds/Sherlock5sec.wav")