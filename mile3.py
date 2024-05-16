import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory of the audio files
directory = 'C:/Users/carli/OneDrive/ATD_23_24/01'
participant = "01"

# Configuration
window_length = 250
overlap = 200
num_participants = 10
num_repetitions = 50

# Arrays to store spectral features
power_med = np.zeros((num_participants, num_repetitions))
power_max = np.zeros((num_participants, num_repetitions))
power_min = np.zeros((num_participants, num_repetitions))
asymmetry = np.zeros((num_participants, num_repetitions))
kurtosis_val = np.zeros((num_participants, num_repetitions))
flatness = np.zeros((num_participants, num_repetitions))
spectral_flux = np.zeros((num_participants, num_repetitions))
spectral_spread = np.zeros((num_participants, num_repetitions))
spectral_crest_factor = np.zeros((num_participants, num_repetitions))
spectral_entropy = np.zeros((num_participants, num_repetitions))

# Loop over audio files
for i in range(num_participants):
    for repetition in range(num_repetitions):
        # Import libraries within the loop
        import os
        import librosa
        from scipy.signal import spectrogram
        from scipy.stats import skew, kurtosis, gmean, entropy

        file_path = os.path.join(directory, f"{i}_{participant}_{repetition}.wav")
        audio_data, freq = librosa.load(file_path, sr=None)

        # Compute spectrogram
        _, F, S = spectrogram(audio_data, fs=freq, window='hann', nperseg=window_length, noverlap=overlap)

        # Update nfft to match the size of the frequency axis (F)
        nfft = len(F)

        power = np.mean(np.abs(S) ** 2, axis=0)

        # Store spectral features
        power_med[i, repetition] = np.mean(power)
        power_max[i, repetition] = np.max(power)
        power_min[i, repetition] = np.min(power)
        asymmetry[i, repetition] = skew(power)
        kurtosis_val[i, repetition] = kurtosis(power)
        flatness[i, repetition] = gmean(power) / np.mean(power)
        spectral_flux[i, repetition] = np.sum(np.diff(power) ** 2)
        normalized_power = power / np.sum(power)
        spectral_entropy[i, repetition] = -np.sum(normalized_power * np.log2(normalized_power))
        spectral_spread[i, repetition] = np.std(power)
        spectral_crest_factor[i, repetition] = np.max(power) / np.mean(power)

# Plot box plots for each spectral feature
spectral_features = {
    "Average Power Spectrum": power_med,
    "Maximum Power Spectrum": power_max,
    "Minimum Power Spectrum": power_min,
    "Asymmetry of Power Distribution": asymmetry,
    "Kurtosis of Power Distribution": kurtosis_val,
    "Flatness of Power Distribution": flatness,
    "Spectral Flux": spectral_flux,
    "Spectral Spread": spectral_spread,
    "Spectral Crest Factor": spectral_crest_factor,
    "Spectral Entropy": spectral_entropy
}

for feature_name, feature_data in spectral_features.items():
    plt.figure(figsize=(8, 6))
    for i in range(num_participants):
        plt.boxplot(feature_data[i])
        plt.xticks([1], [str(i)])
        plt.title(feature_name + ' - Participant ' + str(i))
        plt.xlabel('Repetition')
        plt.ylabel('Value')
        plt.show()

# 3D Scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'brown', 'darkgreen', 'navy', 'gray']

for i in range(num_participants):
    ax.scatter(asymmetry[i], spectral_crest_factor[i], spectral_entropy[i], c=colors[i], label=str(i))

ax.set_xlabel('Asymmetry')
ax.set_ylabel('Crest Factor')
ax.set_zlabel('Entropy')
ax.set_title('3D Scatter Plot of Spectral Features')
plt.legend(title='Digit')

plt.show()
