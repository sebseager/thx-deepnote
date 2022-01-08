import sys
import math
import numpy as np
from numpy import random as random
import scipy.io.wavfile as wavfile
from scipy import signal

# https://www.thx.com/deepnote
# https://en.wikipedia.org/wiki/Deep_Note
# https://pages.mtu.edu/~suits/notefreqs.html

sample_rate = 44100
two_pi = 2 * np.pi

base_freqs = {
    "C": 16.35,
    "C#": 17.32,
    "D": 18.35,
    "D#": 19.45,
    "E": 20.60,
    "F": 21.83,
    "F#": 23.12,
    "G": 24.50,
    "G#": 25.96,
    "A": 27.50,
    "A#": 29.14,
    "B": 30.87,
}

# phase 1
# each voice moves slowly and randomly
# 30 voices at random pitches between 200-400 Hz

random_time = 3
random_min_hz = 200
random_max_hz = 400

# phase 2
# all voices proceed directly to target note

converge_time = 5

# phase 3
# 3 voices per note in treble
# 2 voices per note in bass
# each voice is slightly and randomly detuned

hold_time = 5
n_bass_voices = 2
n_treble_voices = 3
hold_freq_deviation = 0.05
target_chord_bass = [
    ("D", 1),
    ("D", 2),
    ("A", 2),
    ("D", 3),
    ("A", 3),
]
target_chord_treble = [
    ("D", 4),
    ("A", 4),
    ("D", 5),
    ("A", 5),
    ("D", 6),
    ("F#", 6),
]

n_all_voices = (
    len(target_chord_bass) * n_bass_voices + len(target_chord_treble) * n_treble_voices
)

all_voices = [
    (n_bass_voices, target_chord_bass),
    (n_treble_voices, target_chord_treble),
]


def key_to_freq(key, octave):
    return base_freqs[key] * 2 ** octave


def build_wav():
    arr = np.zeros(sample_rate * (random_time + converge_time + hold_time))

    # random phase

    n_samples = sample_rate * random_time
    n_samples_inv = 1 / float(n_samples)
    begin = 0
    end = n_samples
    random_notes = np.zeros(n_all_voices)
    samples = np.arange(n_samples) / float(sample_rate)

    for n_voices, notes in all_voices:
        for v in range(n_voices):
            for i, (key, octave) in enumerate(notes):
                freq = random.randint(random_min_hz, random_max_hz)
                diff_above = random_max_hz - freq
                diff_below = freq - random_min_hz
                if diff_above > diff_below:
                    target_freq = freq + random.randint(0, diff_above)
                else:
                    target_freq = freq + random.randint(0, diff_below)

                # save final freq for phase 2
                random_notes[i * n_voices + v] = target_freq

                # increase amplitude linearly
                amp = np.linspace(0.1, 0.4, num=n_samples, endpoint=True)

                # frequency wanders
                tmp = np.zeros(n_samples)
                for j in range(n_samples):
                    tmp[j] += two_pi * freq * samples[j]
                    freq += (target_freq - freq) * n_samples_inv
                arr[begin:end] += np.sin(tmp) * amp

    # converge phase

    n_samples = sample_rate * converge_time
    n_samples_inv = 1 / float(n_samples)
    begin = end
    end = begin + n_samples
    samples = np.arange(n_samples) / float(sample_rate)

    for n_voices, notes in all_voices:
        for v in range(n_voices):
            for i, (key, octave) in enumerate(notes):
                current_freq = random_notes[i * n_voices + v]
                target_freq = key_to_freq(key, octave)
                # increase amplitude linearly
                amp = np.linspace(0.4, 0.8, num=n_samples, endpoint=True)
                # move towards target note
                tmp = np.zeros(n_samples)
                for j in range(n_samples):
                    tmp[j] += two_pi * current_freq * samples[j]
                    current_freq += (target_freq - current_freq) * n_samples_inv
                arr[begin:end] += np.sin(tmp) * amp

    # hold phase

    n_samples = sample_rate * hold_time
    begin = end
    end = begin + n_samples
    samples = np.arange(n_samples) / float(sample_rate)

    for n_voices, notes in all_voices:
        for v in range(n_voices):
            detune = random.uniform(-hold_freq_deviation, hold_freq_deviation)
            for i, (key, octave) in enumerate(notes):
                freq = key_to_freq(key, octave) + detune
                # increase amplitude linearly
                amp = np.linspace(0.6, 1.0, num=n_samples, endpoint=True)
                arr[begin:end] += amp * np.sin(two_pi * freq * samples)

    # scale output between -1, 1
    # https://stackoverflow.com/a/51085663

    arr_max = np.max(np.abs(arr))
    scaled_arr = (arr / arr_max).astype(np.float32)
    return scaled_arr


if __name__ == "__main__":
    wavfile.write("thx.wav", sample_rate, build_wav())
