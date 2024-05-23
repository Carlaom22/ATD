import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Diretório contendo os arquivos de áudio
directory = 'C:/Users/carli/OneDrive/ATD_23_24/Projeto/01'
participant = "01"
iteration = 0
num_repetitions = 50

# Inicializando listas para armazenar as características de áudio
max_amplitudes = []
min_amplitudes = []
energies = []
average_energies = []
amplitude_ratios = []
durations = []
standard_deviations = []

# Função para processar arquivos de áudio e extrair características
def process_audio_files(directory, participant, iteration):
    for i in range(10):
        print(f"Data for number {i}")
        number = str(i)
        max_amplitude = []
        min_amplitude = []
        amplitude_ratio = []
        energy = []
        duration_list = []
        std_dev_list = []  # Correção aqui para usar uma lista

        for repetition in range(num_repetitions):
            file_path = os.path.join(directory, f"{number}_{participant}_{repetition}.wav")
            audio, sr = librosa.load(file_path, sr=None)

            duration = len(audio) / sr
            std_dev_value = np.std(audio)  # Correção aqui para calcular o valor da desvio padrão

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

            # Calcular a energia 
            e = np.sum(np.abs(audio) ** 2)
            max_amplitude.append(np.max(audio))
            min_amplitude.append(np.min(audio))
            amplitude_ratio.append(np.max(np.abs(audio)) / np.mean(np.abs(audio)))

            # Remover o silêncio inicial
            threshold = np.max(audio) * 0.025
            first_index = np.argmax(audio > threshold)
            audio = audio[first_index:]

            # Ajustar o comprimento do áudio
            num_zeros = sr - len(audio)
            if num_zeros > 0:
                audio = np.pad(audio, (0, num_zeros), 'constant')

            # Normalizar o áudio
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
            std_dev_list.append(std_dev_value)  # Correção aqui para adicionar o valor na lista

        max_amplitudes.append(max_amplitude)
        min_amplitudes.append(min_amplitude)
        amplitude_ratios.append(amplitude_ratio)
        energies.append(energy)
        durations.append(duration_list)
        standard_deviations.append(std_dev_list)  # Correção aqui para usar a lista de desvio padrão

        total_energy = np.sum(energy)
        average_energy_total = total_energy / num_repetitions
        average_energies.append(average_energy_total)

        print(f'The average total energy of the normalized audios without original silence for number {i} is: {average_energy_total:.4f}')
        print(f'The maximum amplitude of the original signal represented is {max_amplitude[iteration]:.4f}')
        print(f'The minimum amplitude of the original signal represented is {min_amplitude[iteration]:.4f}')
        print(f'The amplitude ratio of the original signal represented is {amplitude_ratio[iteration]:.4f}')
        print(f'The standard deviation of the signal is {std_dev_list[iteration]:.4f}')  # Correção aqui para imprimir o valor correto
        print("--------------------------------------------------------")

# Processar os arquivos de áudio
process_audio_files(directory, participant, iteration)

# Converter listas para arrays numpy para plotagem
x = np.arange(10)
max_amplitudes = np.array(max_amplitudes)
energies = np.array(energies)
standard_deviations = np.array(standard_deviations)

# Plotar amplitudes máximas
plt.figure()
plt.plot(x, max_amplitudes, 'o')
plt.xlabel('Numbers')
plt.ylabel('Maximum Amplitudes')
plt.show()

# Plotar energias
plt.figure()
plt.plot(x, energies, 'o')
plt.xlabel('Numbers')
plt.ylabel('Energies')
plt.show()

# Plotar desvios padrão
plt.figure()
plt.plot(x, standard_deviations, 'o')
plt.xlabel('Numbers')
plt.ylabel('Standard Deviations')
plt.show()

# Plot 3D das características
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'gray', 'orange', 'brown', 'k']
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
