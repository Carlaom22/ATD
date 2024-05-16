import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

path = r'C:\Users\carli\OneDrive\ATD_23-24\01'  

# 1) Show one example to see if it´s working
audioExample, samplingRate = librosa.load(os.path.join(path, "0_01_0.wav"), sr=None)
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(audioExample) / samplingRate, len(audioExample)), audioExample)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Exemplo de Sinal de Áudio')
plt.grid(True)
plt.show()

# 2) Calculating temporal characteristics of signals
totalEnergy = []
maxAmplitude = []
minAmplitude = []
rAmplitudes = []
standartDeviation = []

for x in range(10):
    for y in range(50):
        fileName = os.path.join(path, f"{x}_01_{y}.wav")

        # Similar to audioread()
        audio, samplingRate = librosa.load(fileName, sr=None)

        # Pre-processing: removal of initial silence, normalization of amplitude and addition of silence at the end
        audio = librosa.effects.trim(audio)[0]  # Removal of initial silence
        audio = audio / np.max(np.abs(audio))   # Normalization of amplitude

        # Add or remove silence to make them all the same time
        duration = 3  # Seconds
        if len(audio) < samplingRate * duration:
            audio = np.pad(audio, (0, samplingRate * duration - len(audio)), mode='constant')
        elif len(audio) > samplingRate * duration:
            audio = audio[:samplingRate * duration]

        #
        totalEnergy.append(np.sum(audio ** 2))
        maxAmplitude.append(np.max(audio))
        minAmplitude.append(np.min(audio))
        rAmplitudes.append(np.max(audio) / np.abs(np.min(audio)))
        standartDeviation.append(np.std(audio))

#explicar esta parte

       

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.boxplot(totalEnergy)
plt.title('Energia Total')

plt.subplot(3, 2, 2)
plt.boxplot(maxAmplitude)
plt.title('Amplitude Máxima')

plt.subplot(3, 2, 3)
plt.boxplot(minAmplitude)
plt.title('Amplitude Mínima')

plt.subplot(3, 2, 4)
plt.boxplot(rAmplitudes)
plt.title('Razão de Amplitudes')

plt.subplot(3, 2, 5)
plt.boxplot(standartDeviation)
plt.title('Desvio Padrão da Amplitude')

plt.tight_layout()
plt.show()
