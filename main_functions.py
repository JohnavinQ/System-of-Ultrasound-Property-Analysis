import os
import datetime
import numpy as np
import pandas as pd
import serial
import csv
from scipy.signal import butter, filtfilt, hilbert, correlate
from scipy.ndimage import median_filter
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

DATA_DIR = "sweeps"
os.makedirs(DATA_DIR, exist_ok=True)
LOW = 0.8e6
HIGH = 1.2e6
NYQ = 50e6 / 2

B, A = butter(4, [LOW/NYQ, HIGH/NYQ], btype='band')

# ==========================================================
# WATER REFERENCE
# ==========================================================

class WaterReference:
    def __init__(self, csv_path="WaterReferences.csv"):
        df = pd.read_csv(csv_path)

        self.freqs = np.array([float(c.replace("F", "")) for c in df.columns])
        self.data = df.to_numpy()

        #ADD CACHE
        self.cache = {}
        
    def get_signal(self, frequency_mhz):
        key = round(frequency_mhz, 2)

        if key in self.cache:
            return self.cache[key]

        idx = np.argmin(np.abs(self.freqs - key))
        sig = self.data[:, idx]

        self.cache[key] = sig   #store AFTER computing
        return sig


# ==========================================================
# HARDWARE
# ==========================================================

def running_on_pi():
    return os.path.exists("/sys/firmware/devicetree/base/model")



class USARTReader:
    def __init__(self, port="/dev/serial0", baud=115200):
        self.ser = serial.Serial(port, baudrate=baud, timeout=1)

    def _read_byte(self):
        b = self.ser.read(1)
        return b[0] if b else None

    def _find_start(self):
        # look for AA 55
        while True:
            b1 = self._read_byte()
            if b1 is None:
                continue

            if b1 == 0xAA:
                b2 = self._read_byte()
                if b2 == 0x55:
                    return True

    def _wait_end(self):
        # look for 55 AA
        while True:
            b1 = self._read_byte()
            if b1 is None:
                continue

            if b1 == 0x55:
                b2 = self._read_byte()
                if b2 == 0xAA:
                    return True

    def read_packet(self, max_samples=7000):
        data = []

        # 1. wait for frame start
        self._find_start()

        # 2. read samples until frame end or limit
        while len(data) < max_samples:

            # peek ahead for end frame
            self.ser.timeout = 0.001  # fast check
            b1 = self._read_byte()

            if b1 is None:
                continue

            if b1 == 0x55:
                b2 = self._read_byte()
                if b2 == 0xAA:
                    break  # end frame detected

            # otherwise treat as data stream
            b2 = self._read_byte()
            if b2 is None:
                continue

            word = (b1 << 8) | b2

            # extract top 10-bit ADC
            adc = (word >> 6) & 0x03FF

            data.append(adc)

        return np.array(data, dtype=np.uint16)

    def close(self):
        self.ser.close()


def process_usart_data(port="/dev/serial0", Vref=976e-6):
    reader = USARTReader(port)
    try:
        data = reader.read_packet()
    finally:
        reader.close()

    data = data.astype(np.float32)

    offset = 512.0
    return (data - offset) * Vref

# ==========================================================
# PHYSICS ENGINE
# ==========================================================
def hampel_filter(signal, window_size=3, n_sigma=3):
    signal = np.asarray(signal)

    # Rolling median
    med = median_filter(signal, size=2*window_size+1, mode='reflect')

    # Absolute deviation from median
    diff = np.abs(signal - med)

    # Rolling MAD
    mad = median_filter(diff, size=2*window_size+1, mode='reflect')

    # Prevent division issues / zero threshold
    mad[mad == 0] = 1e-8

    threshold = n_sigma * 1.4826 * mad

    # Detect outliers
    outliers = diff > threshold

    # Replace
    cleaned = signal.copy()
    cleaned[outliers] = med[outliers]

    return cleaned


def compute_attenuation(V_s, V_w, fs, f_signal, l):
        # -------- Filtering -------- #


    V_s_cleaned = hampel_filter(V_s)
    V_w_cleaned = hampel_filter(V_w)

    V_w_filtered = filtfilt(B, A, V_w_cleaned)
    V_s_filtered = filtfilt(B, A, V_s_cleaned)

    # -------- Envelope -------- #
    env_w = np.abs(hilbert(V_w_filtered))
    env_s = np.abs(hilbert(V_s_filtered))

    env_w_norm = env_w / np.max(env_w)
    env_s_norm = env_s / np.max(env_s)

    xc = correlate(env_w_norm, env_s_norm, mode='full')
    lags = np.arange(-len(env_s_norm)+1, len(env_w_norm))

    idx = np.argmax(xc)
    delay = lags[idx]

  


    # -------- Windowing -------- #
    thresh = 0.3
    window_w = np.where(env_w_norm >= thresh)[0]
    window_s = window_w-delay

    V_w_windowed = V_w_filtered[window_w]
    s_windowed = V_s_filtered[window_s]

    # -------- Energy -------- #
    Ew = trapezoid(V_w_windowed**2)
    Es = trapezoid(s_windowed**2)

    attenuation = -(10 / l) * np.log10(Es / Ew) / 100  # dB/cm

    return attenuation, s_windowed

def enforce_sweep_limit(limit=100):
    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    ]

    if len(files) <= limit:
        return

    # oldest first
    files.sort(key=os.path.getctime)

    for f in files[:len(files) - limit]:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Delete failed: {f} -> {e}")
# ==========================================================
# SWEEP SESSION
# ==========================================================

class SweepSession:
    def __init__(self, water_ref):
        now = datetime.datetime.now()

        self.sweep_id = now.strftime("test_%Y-%m-%d_%H-%M-%S")
        self.created = now.isoformat()

        self.water_ref = water_ref
        self.frequencies = []
        self.attenuations = []
        self.finalized = False
        self.coefficient = None
        self.windowed_signals = []

    def run_test(self, freq_mhz, V_S, fs=50e6, l=0.02):

        V_W = self.water_ref.get_signal(freq_mhz)

        if len(V_S) == 0 or len(V_W) == 0:
            raise ValueError("Empty signal")

        n = min(len(V_S), len(V_W))
        V_S, V_W = V_S[:n], V_W[:n]

        return compute_attenuation(V_S, V_W, fs, freq_mhz*1e6, l)

    def add_frequency(self, freq_mhz, V_S):
        if self.finalized:
            raise Exception("Finalized sweep")

        if V_S is None or len(V_S) < 100:
            raise Exception("Invalid signal")

        att, windowed = self.run_test(freq_mhz, V_S)

        self.frequencies.append(freq_mhz)
        self.attenuations.append(att)
        self.windowed_signals.append(windowed)

    def finalize(self):
        if len(self.frequencies) < 2:
            raise Exception("Need more points")

        m, b = np.polyfit(self.frequencies, self.attenuations, 1)

        self.coefficient = m
        self.finalized = True

    def save(self):
        path = os.path.join(DATA_DIR, f"{self.sweep_id}.csv")

        # ensure coefficient exists
        if not self.finalized:
            self.finalize()

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # ===== GLOBAL INFO =====
            writer.writerow(["coefficient", self.coefficient])
            writer.writerow(["sweep_id", self.sweep_id])
            writer.writerow(["created", self.created])
            writer.writerow([])

            # ===== PER SAMPLE BLOCK =====
            for i in range(len(self.frequencies)):
                writer.writerow(["sample", i + 1])
                writer.writerow(["frequency_mhz", self.frequencies[i]])
                writer.writerow(["attenuation_db", self.attenuations[i]])

                # voltage row (all values in one row)
                voltage_row = ["voltage"] + list(self.windowed_signals[i])
                writer.writerow(voltage_row)

                writer.writerow([])  # spacing between samples

        enforce_sweep_limit(100)

    @classmethod
    def load(cls, path, water_ref):
        import csv
        import numpy as np

        s = cls(water_ref)

        frequencies = []
        attenuations = []
        coefficient = None

        with open(path, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row:
                    continue

                # GLOBAL HEADER
                if row[0] == "coefficient":
                    try:
                        coefficient = float(row[1])
                    except:
                        coefficient = None

                # DATA ROWS
                elif row[0] == "frequency_mhz":
                    try:
                        freq = float(row[1])
                        frequencies.append(freq)
                    except:
                        pass

                elif row[0] == "attenuation_db":
                    try:
                        att = float(row[1])
                        attenuations.append(att)
                    except:
                        pass

        s.coefficient = coefficient
        s.frequencies = frequencies
        s.attenuations = attenuations
        s.finalized = True

        return s