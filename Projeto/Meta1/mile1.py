import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the audio files
directory = 'C:/Users/carli/OneDrive/ATD_23_24/Projeto/01'
participant = "01"
iteration = 0
num_repetitions = 50

# Initializing lists to store audio features
max_amplitudes = []
min_amplitudes = []
energies = []
average_energies = []
amplitude_ratios = []
durations = []
standard_deviations = []

# Function to process audio files and extract features
def process_audio_files(directory, participant, iteration):
    for i in range(10):
        print(f"Data for number {i}")
        number = str(i)
        max_amplitude = []
        min_amplitude = []
        amplitude_ratio = []
        energy = []
        duration_list = []
        std_dev = []

        for repetition in range(num_repetitions):
            # Construct the file path
            file_path = os.path.join(directory, f"{number}_{participant}_{repetition}.wav")
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=None)

            # Calculate the duration of the original audio
            duration = len(audio) / sr
            std_dev = np.std(audio)

            if repetition == iteration:
                namp = len(audio)
                time = np.arange(namp) * (1 / sr)
                plt.figure()
                plt.plot(time, audio)
                plt.xlabel("Time[s]")
                plt.ylabel("Amplitude")
                plt.title(f"{number}", fontweight="bold")
                plt.show()
                print(f"The audio has a duration of {duration:.4f} seconds")

            # Calculate the energy of the audio
            e = np.sum(np.abs(audio) ** 2)
            max_amplitude.append(np.max(audio))
            min_amplitude.append(np.min(audio))
            amplitude_ratio.append(np.max(np.abs(audio)) / np.mean(np.abs(audio)))

            # Remove initial silence
            threshold = np.max(audio) * 0.025
            first_index = np.argmax(audio > threshold)
            audio = audio[first_index:]

            # Adjust the length of the audio
            num_zeros = sr - len(audio)
            if num_zeros > 0:
                audio = np.pad(audio, (0, num_zeros), 'constant')

            # Normalize the audio
            audio = -1 + 2 * (audio - np.min(audio)) / (np.max(audio) - np.min(audio))

            if repetition == iteration:
                time = np.arange(len(audio)) * (1 / sr)
                plt.figure()
                plt.plot(time, audio)
                plt.xlabel("Time[s]")
                plt.ylabel("Amplitude")
                plt.title(f"{number} without initial silence")
                plt.show()

            energy.append(e)
            duration_list.append(duration)
            std_dev.append(std_dev)

        max_amplitudes.append(max_amplitude)
        min_amplitudes.append(min_amplitude)
        amplitude_ratios.append(amplitude_ratio)
        energies.append(energy)
        durations.append(duration_list)
        standard_deviations.append(std_dev)

        total_energy = np.sum(energy)
        average_energy_total = total_energy / 50
        average_energies.append(average_energy_total)

        print(f'The average total energy of the normalized audios without original silence for number {i} is: {average_energy_total:.4f}')
        print(f'The maximum amplitude of the original signal represented is {max_amplitude[iteration]:.4f}')
        print(f'The minimum amplitude of the original signal represented is {min_amplitude[iteration]:.4f}')
        print(f'The amplitude ratio of the original signal represented is {amplitude_ratio[iteration]:.4f}')
        print(f'The standard deviation of the signal is {std_dev[iteration]:.4f}')
        print("--------------------------------------------------------")

# Process the audio files
process_audio_files(directory, participant, iteration)

# Convert lists to numpy arrays for plotting
x = np.arange(10)
max_amplitudes = np.array(max_amplitudes)
energies = np.array(energies)
standard_deviations = np.array(standard_deviations)

# Plot maximum amplitudes
plt.figure()
plt.plot(x, max_amplitudes, 'o')
plt.xlabel('Numbers')
plt.ylabel('Maximum Amplitudes')
plt.show()

# Plot energies
plt.figure()
plt.plot(x, energies, 'o')
plt.xlabel('Numbers')
plt.ylabel('Energies')
plt.show()

# Plot standard deviations
plt.figure()
plt.plot(x, standard_deviations, 'o')
plt.xlabel('Numbers')
plt.ylabel('Standard Deviations')
plt.show()

# 3D plot of the features
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'gray', 'brown', 'k', 'orange']
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    ax.scatter(max_amplitudes[i], energies[i], standard_deviations[i], c=colors[i], label=labels[i])

ax.set_xlabel('Maximum Amplitudes')
ax.set_ylabel('Energies')
ax.set_zlabel('Standard Deviations')
ax.legend()
plt.show()
