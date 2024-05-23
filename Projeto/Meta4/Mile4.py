import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import librosa
from scipy.signal import spectrogram
from scipy.stats import skew, entropy

# Directory of the audio files
directory = 'C:/Users/carli/OneDrive/ATD_23_24/Projeto/01'
participant = "01"

# Configuration
window_length = 250
overlap = 200
num_participants = 10
num_repetitions = 50

# Arrays to store spectral features
amplitude_max = np.zeros((num_participants, num_repetitions))
energy = np.zeros((num_participants, num_repetitions))
std_dev = np.zeros((num_participants, num_repetitions))
spectral_contrast = np.zeros((num_participants, num_repetitions))
entropy_val = np.zeros((num_participants, num_repetitions))
spectral_entropy = np.zeros((num_participants, num_repetitions))
power_contrast = np.zeros((num_participants, num_repetitions))
asymmetry = np.zeros((num_participants, num_repetitions))

# Small constant to avoid log(0)
epsilon = 1e-10

# Loop over audio files
for i in range(num_participants):
    for repetition in range(num_repetitions):
        file_path = os.path.join(directory, f"{i}_{participant}_{repetition}.wav")
        audio_data, freq = librosa.load(file_path, sr=None)

        # Compute spectrogram
        _, _, S = spectrogram(audio_data, fs=freq, window='hann', nperseg=window_length, noverlap=overlap)

        power = np.mean(np.abs(S) ** 2, axis=0)

        # Store spectral features
        amplitude_max[i, repetition] = np.max(audio_data)
        energy[i, repetition] = np.sum(np.abs(audio_data) ** 2)
        std_dev[i, repetition] = np.std(audio_data)
        spectral_contrast[i, repetition] = np.std(np.diff(np.log(np.abs(S).flatten() + epsilon)))
        entropy_val[i, repetition] = -np.sum(np.abs(audio_data) * np.log2(np.abs(audio_data) + epsilon))
        normalized_power = power / np.sum(power + epsilon)
        spectral_entropy[i, repetition] = -np.sum(normalized_power * np.log2(normalized_power + epsilon))
        power_contrast[i, repetition] = np.std(np.diff(np.log(power + epsilon)))
        asymmetry[i, repetition] = skew(power)

# Classification function
def classify_digit(amplitude_max, energy, std_dev, spectral_contrast, entropy_val, spectral_entropy, power_contrast, asymmetry):
    if 0.0015 < std_dev < 0.005:
        if 0.5 < energy < 0.6:
            return 1
        elif 0.3 < energy < 0.4:
            return 3
    elif 0.2 < energy < 0.6:
        return 6
    elif 7.8 < spectral_entropy < 8.3:
        if 2500 < spectral_contrast < 3850:
            return 9
        else:
            return 0
    elif 0.300 < spectral_contrast < 0.400:
        if 1.2e-5 < spectral_entropy < 1.5e-5:
            return 9
        else:
            return 1
    elif 0.160 < spectral_contrast < 0.260:
        if 300 < entropy_val < 550:
            return 5
        elif 300 < entropy_val < 700:
            return 8
        elif 400 < entropy_val < 500:
            if 0.6e-6 < spectral_entropy < 2.0e-5:
                return 4
            elif 0.3e-6 < spectral_entropy < 1.7e-5:
                return 7
    elif 0.0060 < amplitude_max < 0.0189:
        if 0.160 < spectral_contrast < 0.240:
            return 7
    return -1  # No match found

# Classify all samples and calculate accuracy
correct_count = 0
total_samples = num_participants * num_repetitions

for i in range(num_participants):
    for repetition in range(num_repetitions):
        predicted_digit = classify_digit(amplitude_max[i, repetition], energy[i, repetition], std_dev[i, repetition], 
                                         spectral_contrast[i, repetition], entropy_val[i, repetition], 
                                         spectral_entropy[i, repetition], power_contrast[i, repetition], 
                                         asymmetry[i, repetition])
        if predicted_digit == i:
            correct_count += 1

accuracy = (correct_count / total_samples) * 100
print(f"Accuracy: {accuracy:.2f}%")
print("Correct count:",correct_count)

# Plotting the results
features = {
    "Maximum Amplitude": amplitude_max,
    "Energy": energy,
    "Standard Deviation": std_dev,
    "Spectral Contrast": spectral_contrast,
    "Entropy": entropy_val,
    "Spectral Entropy": spectral_entropy,
    "Power Contrast": power_contrast,
    "Asymmetry of Power Distribution": asymmetry
}

for feature_name, feature_data in features.items():
    plt.figure()
    plt.boxplot(feature_data.T)
    plt.title(feature_name)
    plt.xlabel('Participant')
    plt.ylabel('Feature Value')
    plt.show()

# 3D Scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'brown', 'darkgreen', 'navy', 'gray']

for i in range(num_participants):
    ax.scatter(asymmetry[i], power_contrast[i], spectral_entropy[i], c=colors[i], label=str(i))

ax.set_xlabel('Asymmetry')
ax.set_ylabel('Power Contrast')
ax.set_zlabel('Spectral Entropy')
ax.set_title('3D Scatter Plot of Spectral Features')
plt.legend(title='Digit')
plt.show()
