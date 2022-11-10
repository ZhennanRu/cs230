import numpy as np
import matplotlib.pyplot as plt
import librosa
# load wave file and sampling frequency
y, sr = librosa.load('../voice/tsuchiya_normal/tsuchiya_normal_001.wav')

plt.plot(y)
