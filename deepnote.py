import sys
import math
import numpy as np
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

random_time = 4
random_start_amp = 0.1
random_min_hz = 200
random_max_hz = 400

# phase 2
# all voices proceed directly to target note

converge_time = 5
converge_start_amp = 0.5

# phase 3
# 3 voices per note in treble
# 2 voices per note in bass
# each voice is slightly and randomly detuned

hold_time = 5
hold_start_amp = 0.9
hold_end_amp = 1.0
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
    begin = 0
    end = n_samples
    random_notes = np.zeros(n_all_voices)
    samples = np.arange(n_samples) / float(sample_rate)
    for n_voices, notes in all_voices:
        for v in range(n_voices):
            for i, (key, octave) in enumerate(notes):
                freq = np.random.randint(random_min_hz, random_max_hz)
                # increase amplitude linearly
                amp = np.linspace(
                    random_start_amp, converge_start_amp, num=n_samples, endpoint=True
                )
                arr[begin:end] += amp * np.sin(two_pi * freq * samples)
                # save final freq for phase 2
                random_notes[v] = freq

    # n_samples = sample_rate * random_time
    # begin = 0
    # end = n_samples
    # random_notes = np.zeros(n_all_voices)
    # samples = np.arange(n_samples) / float(sample_rate)
    # for v in range(n_all_voices):
    #     freq = np.random.randint(random_min_hz, random_max_hz)
    #     # increase amplitude linearly
    #     amp = np.linspace(
    #         random_start_amp, converge_start_amp, num=n_samples, endpoint=True
    #     )
    #     arr[begin:end] += amp * np.sin(two_pi * freq * samples)
    #     # save final freq for phase 2
    #     random_notes[v] = freq

    # converge phase

    n_samples = sample_rate * converge_time
    begin = end
    end = begin + n_samples
    target_notes = np.zeros(n_all_voices)
    samples = np.arange(n_samples) / float(sample_rate)
    for v in range(n_all_voices):
        current_freq = random_notes[v]
        # of every five voices, three should be in treble, two in bass
        target_staff = (
            target_chord_bass
            if v % (n_bass_voices + n_treble_voices) < n_bass_voices
            else target_chord_treble
        )
        target_freq = key_to_freq(*target_staff[v % len(target_staff)])
        target_notes[v] = target_freq

    random_notes = np.sort(random_notes)
    target_notes = np.sort(target_notes)
    print(list(zip(random_notes, target_notes)))
    for v in range(n_all_voices):
        # increase amplitude linearly
        amp = np.linspace(
            converge_start_amp, hold_start_amp, num=n_samples, endpoint=True
        )
        # move linearly towards target note
        arr[begin:end] += amp * np.sin(
            two_pi
            * np.linspace(
                random_notes[v], target_notes[v], num=n_samples, endpoint=True
            )
            * samples
        )

    # hold phase

    n_samples = sample_rate * hold_time
    begin = end
    end = begin + n_samples
    samples = np.arange(n_samples) / float(sample_rate)
    for n_voices, notes in (
        (n_bass_voices, target_chord_bass),
        (n_treble_voices, target_chord_treble),
    ):
        for v in range(n_voices):
            detune = np.random.uniform(-hold_freq_deviation, hold_freq_deviation)
            for i, (key, octave) in enumerate(notes):
                freq = key_to_freq(key, octave) + detune
                # increase amplitude linearly
                amp = np.linspace(hold_start_amp, hold_end_amp, n_samples)
                arr[begin:end] += amp * np.sin(two_pi * freq * samples)

    # scale output between -1, 1
    # https://stackoverflow.com/a/51085663

    arr_max = np.max(np.abs(arr))
    scaled_arr = (arr / arr_max).astype(np.float32)
    return scaled_arr


if __name__ == "__main__":
    wavfile.write("thx.wav", sample_rate, build_wav())
