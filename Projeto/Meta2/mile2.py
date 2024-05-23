import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read and preprocess audio files
def read_audio(directory, participant, digit, repetition):
    filename = os.path.join(directory, f'{digit}_{participant}_{repetition}.wav')
    if os.path.isfile(filename):
        Fs, data = wav.read(filename)
        data = data.astype(float)
        a = np.argmax(data > (np.max(data) * 0.025))
        data = data[a:]
        if len(data) < Fs:
            data = np.pad(data, (0, Fs - len(data)), 'constant')
        return data, Fs
    else:
        print(f'File not found: {filename}')
        return None, None

# Function to calculate the FFT and normalize it
def calculate_fft(data, window):
    windowed_data = data * window
    fft_data = np.abs(np.fft.fftshift(np.fft.fft(windowed_data)))
    fft_data = fft_data[len(fft_data)//2:]
    fft_data /= len(fft_data)
    return fft_data

# Directory containing audio files
directory = "C:/Users/carli/OneDrive/ATD_23_24/Projeto/01"
participant = "01"
num_digits = 10
num_repetitions = 50

# Initialize arrays to store results
N = 48000
frequencies = np.linspace(0, 24000, N//2)
max_amplitude = np.zeros((num_digits, num_repetitions))
max_position = np.zeros((num_digits, num_repetitions))
spectral_mean = np.zeros((num_digits, num_repetitions))
entropies = np.zeros((num_digits, num_repetitions))
peak_frequencies = np.zeros((num_digits, num_repetitions))
peak_magnitudes = np.zeros((num_digits, num_repetitions))
spectral_contrast = np.zeros((num_digits, num_repetitions))
relative_power = np.zeros((num_digits, num_repetitions))

# Median spectrum arrays for different windows
meds_normal = np.zeros((num_digits, N//2))
meds_flattop = np.zeros((num_digits, N//2))
meds_blackman = np.zeros((num_digits, N//2))
meds_hamming = np.zeros((num_digits, N//2))

# Quartile arrays for different windows
q25_normal = np.zeros((num_digits, N//2))
q75_normal = np.zeros((num_digits, N//2))
q25_flattop = np.zeros((num_digits, N//2))
q75_flattop = np.zeros((num_digits, N//2))
q25_blackman = np.zeros((num_digits, N//2))
q75_blackman = np.zeros((num_digits, N//2))
q25_hamming = np.zeros((num_digits, N//2))
q75_hamming = np.zeros((num_digits, N//2))

# Process each digit
for digit in range(num_digits):
    dft_data_normal = np.zeros((num_repetitions, N//2))
    dft_data_flattop = np.zeros((num_repetitions, N//2))
    dft_data_blackman = np.zeros((num_repetitions, N//2))
    dft_data_hamming = np.zeros((num_repetitions, N//2))
    
    for repetition in range(num_repetitions):
        data, Fs = read_audio(directory, participant, digit, repetition)
        if data is not None:
            N = len(data)
            freqs = np.linspace(0, Fs/2, N//2)
            window_normal = np.ones(N)
            window_flattop = signal.windows.flattop(N)
            window_blackman = signal.windows.blackman(N)
            window_hamming = signal.windows.hamming(N)
            
            dft_audio_normal = calculate_fft(data, window_normal)
            dft_audio_flattop = calculate_fft(data, window_flattop)
            dft_audio_blackman = calculate_fft(data, window_blackman)
            dft_audio_hamming = calculate_fft(data, window_hamming)
            
            dft_data_normal[repetition, :] = dft_audio_normal[:N//2]
            dft_data_flattop[repetition, :] = dft_audio_flattop[:N//2]
            dft_data_blackman[repetition, :] = dft_audio_blackman[:N//2]
            dft_data_hamming[repetition, :] = dft_audio_hamming[:N//2]
            
            max_amplitude[digit, repetition] = np.max(dft_audio_normal)
            max_position[digit, repetition] = np.argmax(dft_audio_normal)
            spectral_mean[digit, repetition] = np.mean(dft_audio_normal)
            entropies[digit, repetition] = -np.sum(dft_audio_normal * np.log2(dft_audio_normal + 1e-12))
            total_power = np.sum(dft_audio_normal)
            relative_power[digit, repetition] = np.max(dft_audio_normal) / total_power
            spectral_contrast[digit, repetition] = np.std(np.diff(np.log(dft_audio_normal + 1e-12)))
    
    meds_normal[digit, :] = np.median(dft_data_normal, axis=0)
    meds_flattop[digit, :] = np.median(dft_data_flattop, axis=0)
    meds_blackman[digit, :] = np.median(dft_data_blackman, axis=0)
    meds_hamming[digit, :] = np.median(dft_data_hamming, axis=0)
    
    q25_normal[digit, :] = np.percentile(dft_data_normal, 25, axis=0)
    q75_normal[digit, :] = np.percentile(dft_data_normal, 75, axis=0)
    q25_flattop[digit, :] = np.percentile(dft_data_flattop, 25, axis=0)
    q75_flattop[digit, :] = np.percentile(dft_data_flattop, 75, axis=0)
    q25_blackman[digit, :] = np.percentile(dft_data_blackman, 25, axis=0)
    q75_blackman[digit, :] = np.percentile(dft_data_blackman, 75, axis=0)
    q25_hamming[digit, :] = np.percentile(dft_data_hamming, 25, axis=0)
    q75_hamming[digit, :] = np.percentile(dft_data_hamming, 75, axis=0)

# Plotting functions
def plot_spectrum(frequencies, q25, median, q75, title):
    plt.figure(figsize=(18, 12))
   
    for i in range(num_digits):
        plt.subplot(5, 2, i + 1)
        plt.plot(frequencies, q75[i, :], label='Q75', linewidth=0.5, color='green')
        plt.plot(frequencies, median[i, :], label='Median', linewidth=0.5, color='blue')
        plt.plot(frequencies, q25[i, :], label='Q25', linewidth=0.5, color='red')
        plt.xlim([0, 1000])
        plt.title(f'Digit {i}')
        plt.legend()
    plt.suptitle(title)
    plt.show()

plot_spectrum(frequencies, q25_normal, meds_normal, q75_normal, 'Normal Window')
plot_spectrum(frequencies, q25_flattop, meds_flattop, q75_flattop, 'Flat Top Window')
plot_spectrum(frequencies, q25_blackman, meds_blackman, q75_blackman, 'Blackman Window')
plot_spectrum(frequencies, q25_hamming, meds_hamming, q75_hamming, 'Hamming Window')

# Boxplots for entropies, and spectral contrast
def plot_boxplot(data, title, xlabel, ylabel):
    plt.figure()
    plt.boxplot(data.T, labels=[str(i) for i in range(num_digits)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

plot_boxplot(entropies, 'Spectral Entropies', 'Digit', 'Entropy')
plot_boxplot(spectral_contrast, 'Spectral Contrast', 'Digit', 'Spectral Contrast')

# 3D Scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.tab10.colors

for i in range(num_digits):
    ax.scatter(entropies[i, :], spectral_contrast[i, :], relative_power[i, :], color=colors[i], label=f'Digit {i}')

ax.set_xlabel('Spectral Entropy')
ax.set_ylabel('Spectral Contrast')
ax.set_zlabel('Relative Power')
ax.set_title('3D Scatter Plot')
plt.legend()
plt.show()
