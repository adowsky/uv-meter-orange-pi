import numpy as np
import glob
import struct
import pyaudio
import wave
import time
from collections import namedtuple
import threading

audio = pyaudio.PyAudio()
window_fun = np.hamming

sound_thread = threading.Thread()
current_chunk = []
CHUNK_SIZE = 2048
FFT_SIZE = 200
MAX_SHORT = 32768
WINDOW = window_fun(CHUNK_SIZE)
time_iter = []
Chunk = namedtuple("Chunk", ["raw_data", "processed_data"])


class Player:
    def __init__(self, wave_samplewidth, wave_channels, wave_rate, initial_buffer_size, processed_data_handler):
        self._stream = audio.open(format=pyaudio.get_format_from_width(wave_samplewidth), channels=wave_channels,
                                  rate=wave_rate, frames_per_buffer=CHUNK_SIZE, output=True,
                                  stream_callback=self.callback)
        self._buffer_size = initial_buffer_size
        self._processed_data_handler = processed_data_handler
        self._play_lock = threading.Lock()
        self._add_lock = threading.Lock()
        self._play_lock.acquire()
        self._stream.start_stream()
        self._buffer = []
        self._finish = False

    def callback(self, in_data, frame_count, time_info, status):
        data = self._get_next_chunk()
        self._processed_data_handler(data.processed_data)
        status = pyaudio.paComplete if self._finish else pyaudio.paContinue
        return data.raw_data, status

    def _get_next_chunk(self):
        self._add_lock.acquire()
        try:
            chunk = self._buffer.pop(0) if len(self._buffer) > 0 else Chunk("", None)
        finally:
            self._add_lock.release()
        return chunk

    def add_chunk(self, chunk):
        self._add_lock.acquire()
        try:
            self._buffer.append(chunk)
        finally:
            self._add_lock.release()

    def shutdown(self):
        self._finish = True
        self._stream.stop_stream()
        self._stream.close()
        audio.terminate()

    def wait_for_finish(self):
        while self._stream.is_active():
            time.sleep(0.1)


def reduce_to_single_channel_avg(chunk, channels):
    return [np.mean(chunk[i:i + channels]) for i in range(0, len(chunk), channels)]


class Song:
    def __init__(self, wavefile):
        self._wave_file = wave.open(wavefile, 'rb')
        self.rate = self._wave_file.getframerate()
        self.channels_count = self._wave_file.getnchannels()
        self.samplewidth = self._wave_file.getsampwidth()
        self.song_length_frames = self._wave_file.getnframes()
        self._elapsed_frames = 0
        self.freqs_in_fft = np.fft.rfftfreq(FFT_SIZE, d=1. / self.rate)

    def get_next_chunk(self):
        remaining_frames = self.song_length_frames - self._elapsed_frames
        size_to_get = CHUNK_SIZE if remaining_frames >= CHUNK_SIZE else remaining_frames
        self._elapsed_frames += size_to_get
        return size_to_get, self._wave_file.readframes(size_to_get)

    def max_freq_value(self, freq, spectrum):
        val = -MAX_SHORT
        for i in range(self.find_freq_best_fit_idx(freq[0]),
                       self.find_freq_best_fit_idx(freq[1])):
            val = max(val, spectrum[i])
        return val

    def find_freq_best_fit_idx(self, wanted_frequency):
        best_difference = MAX_SHORT
        best_idx = 0
        for (idx, fft_frequency) in enumerate(self.freqs_in_fft):
            current_difference = abs(fft_frequency - wanted_frequency)
            if current_difference < best_difference:
                best_idx = idx
                best_difference = current_difference
        return best_idx


def compute_spectrum(frames, channels_count):
    start = time.time()
    channel_reduced_chunk = reduce_to_single_channel_avg(frames, channels_count)
    time_iter.append(time.time() - start)

    window = WINDOW if len(WINDOW) == len(channel_reduced_chunk) else window_fun(len(channel_reduced_chunk))
    channel_reduced_chunk *= window
    spectrum = np.fft.rfft(channel_reduced_chunk, FFT_SIZE) * 2 / len(channel_reduced_chunk)
    return np.abs(spectrum)


def handle_measured_frequency_buckets(values):
    # todo leds imlementation
    # every value is in range [0, MAX_SHORT)
    if values is not None and len(values) == 3:

        val = int(values[0]) / int((MAX_SHORT / 3))
        print("bass=%f mid=%f treble=%f " % (
            values[0], values[1], values[2],))


def play(filename):
    song = Song(filename)
    player = Player(song.samplewidth, song.channels_count, song.rate, 2, handle_measured_frequency_buckets)

    chunk_length, data = song.get_next_chunk()
    while len(data) > 0:
        frames = np.fromstring(data, dtype=np.ushort)
        spectrum = compute_spectrum(frames, song.channels_count)
        sound_measures = [
            song.max_freq_value(freq=(50, 300), spectrum=spectrum),
            song.max_freq_value(freq=(310, 5000), spectrum=spectrum),
            song.max_freq_value(freq=(5010, 12000), spectrum=spectrum)
        ]
        player.add_chunk(Chunk(raw_data=data, processed_data=sound_measures))
        # todo for debug
        # print("bass=%f mid=%f treble=%f max=%f" % (sound_measures[0], sound_measures[1], sound_measures[2], spectrum.max() / MAX_SHORT))
        chunk_length, data = song.get_next_chunk()
    player.wait_for_finish()
    player.shutdown()
    print("Fin.")


# Analizator widma -> tytul proektu

# in root directory needs to be at least one wave file
if __name__ == "__main__":
    sound_files = glob.glob("song.wav")
    if len(sound_files) == 0:
        raise Exception("No sound files (.wav) in current directory")

    for sound_file in sound_files:
        play(sound_file)

    print np.mean(time_iter)
