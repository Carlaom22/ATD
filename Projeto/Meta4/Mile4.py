import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import spectrogram
from scipy.stats import skew

# Define directory and participant
directory = 'C:/Users/carli/OneDrive/ATD_23_24/Projeto/01'
participant = "01"
num_participants = 10
num_repetitions = 50


# Initialize arrays
power_contrast = np.zeros((10, 50))
asymmetry = np.zeros((10, 50))
spectral_entropy = np.zeros((10, 50))
entropies = np.zeros((10, 50))
SEFS = np.zeros((10, 50))
spectral_contrast = np.zeros((10, 50))
amplitudes_maximas = np.zeros((10, 50))
energies = np.zeros((10, 50))
std_devs = np.zeros((10, 50))

# Configuration parameters
window_length = 250
overlap = 200
nfft = 1024

# Small constant to avoid log(0)
epsilon = 1e-10

# Loop over audio files
for i in range(num_participants):
    for repetition in range(num_repetitions):
        file_path = os.path.join(directory, f"{i}_{participant}_{repetition}.wav")
        audio_data, freq = librosa.load(file_path, sr=None)
        
        # Compute spectrogram
        _, _, S = spectrogram(audio_data, fs=freq, window='hann', nperseg=window_length, noverlap=overlap, nfft=nfft)
        power = np.mean(np.abs(S) ** 2, axis=0)
        
        # Meta 1 variables
        std_devs[i, repetition] = np.std(audio_data)
        energies[i, repetition] = np.sum(np.abs(audio_data) ** 2)
        amplitudes_maximas[i, repetition] = np.max(audio_data)
        
        # Meta 2 variables
        a = np.argmax(audio_data > (np.max(audio_data) * 0.025))
        audio_data = audio_data[a:]
        if len(audio_data) < freq:
            audio_data = np.pad(audio_data, (0, freq - len(audio_data)), 'constant')
        
        dft_audio_normal = np.fft.fftshift(np.fft.fft(audio_data))
        mid = len(dft_audio_normal) // 2
        dft_audio_normal = np.abs(dft_audio_normal[mid:]) / len(dft_audio_normal)
        
        spectral_contrast[i, repetition] = np.std(np.diff(np.log(dft_audio_normal + epsilon)))
        accumulated_energy = np.cumsum(dft_audio_normal)
        sef90_index = np.argmax(accumulated_energy >= 0.9 * accumulated_energy[-1])
        SEFS[i, repetition] = sef90_index * (freq / len(audio_data))
        entropies[i, repetition] = -np.sum(dft_audio_normal * np.log2(dft_audio_normal + epsilon))
        
        # Meta 3 variables
        asymmetry[i, repetition] = skew(power)
        normalized_power = power / np.sum(power + epsilon)
        spectral_entropy[i, repetition] = -np.sum(normalized_power * np.log2(normalized_power + epsilon))
        power_contrast[i, repetition] = np.std(np.diff(np.log(power + epsilon)))

# Plotting the boxplots
labels = [str(i) for i in range(10)]

def plot_boxplot(data, title, ylabel):
    plt.figure()
    plt.boxplot(data.T)
    plt.xticks(range(1, 11), labels)
    plt.title(title)
    plt.xlabel('Digit')
    plt.ylabel(ylabel)
    plt.show()

plot_boxplot(amplitudes_maximas, 'Maximum Amplitudes of Audio (Meta 1)', 'Maximum Amplitude')
plot_boxplot(energies, 'Energy of Audio (Meta 1)', 'Energy')
plot_boxplot(std_devs, 'Standard Deviation of Audio (Meta 1)', 'Standard Deviation')
plot_boxplot(spectral_contrast, 'Spectral Contrast (Meta 2)', 'Spectral Contrast')
plot_boxplot(entropies, 'Spectral Entropy (Meta 2)', 'Spectral Entropy')
plot_boxplot(SEFS, 'SEFS of Spectrum (Meta 2)', 'SEFS')
plot_boxplot(power_contrast, 'Power Contrast (Meta 3)', 'Power Contrast')
plot_boxplot(spectral_entropy, 'Power Entropy (Meta 3)', 'Power Entropy')
plot_boxplot(asymmetry, 'Power Distribution Asymmetry (Meta 3)', 'Asymmetry')

# Classification function
def classify_digit(spectral_contrast, asymmetry, spectral_entropy, SEFS, entropies, amplitudes_maximas):
    if 0.4 < spectral_contrast < 0.6:
        if 3.3 < asymmetry < 4.5:
            return 2
        elif 2 < asymmetry < 3.3:
            return 3
    elif 4.45 < spectral_entropy < 6.6:
        return 6
    elif 7.8 < spectral_entropy < 8.3:
        if 2500 < SEFS < 3850:
            return 9
        else:
            return 0
    elif 2200 < SEFS < 3700:
        if 7.85 < spectral_entropy < 8.3:
            return 9
        else:
            return 1
    elif 7500 < SEFS < 11000:
        if 1.9 < entropies < 2.6:
            return 5
        elif 1.4 < entropies < 1.75:
            return 8
        elif 1.6 < entropies < 2:
            if 7.4 < spectral_entropy < 7.85:
                return 4
            elif 7.85 < spectral_entropy < 8.3:
                return 0
    elif 0.013 < amplitudes_maximas < 0.02:
        if 9800 < SEFS < 11000:
            return 7
        elif 2200 < SEFS < 3700:
            return 1
        elif 7500 < SEFS < 11000:
            return 5
    return -1

# Classify all samples and calculate accuracy
correct_count = 0
total_samples = 10 * 50

for i in range(10):
    for j in range(50):
        predicted_digit = classify_digit(spectral_contrast[i, j], asymmetry[i, j], spectral_entropy[i, j], SEFS[i, j], entropies[i, j], amplitudes_maximas[i, j])
        if predicted_digit == i:
            correct_count += 1

print(f"Correct classifications: {correct_count}")
accuracy = (correct_count / total_samples) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Plotting 3D scatter plot
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'brown', 'darkgreen', 'navy', 'gray']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    ax.scatter(std_devs[i], SEFS[i], spectral_entropy[i], c=colors[i], label=labels[i])
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('SEFS')
ax.set_zlabel('Power Entropy')
ax.set_title('3D Scatter Plot of Spectral Features')
ax.legend(title='Digit', loc='upper right')
plt.show()
