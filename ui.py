import sys
import os
import numpy as np
import subprocess
import csv

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QLabel, QFrame
)
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject, Qt

from main_functions import SweepSession, WaterReference, DATA_DIR, running_on_pi, compute_attenuation, process_usart_data

# ==========================================================
# GPIO SETUP
# ==========================================================

GPIO_UP = GPIO_DOWN = GPIO_ENTER = None
GPIO_EXPORT = GPIO_SHUTDOWN = GPIO_FREQ = None


# ==========================================================
# USB DETECTION
# ==========================================================

def find_usb():
    base = "/media/pi"
    if os.path.exists(base):
        for d in os.listdir(base):
            return os.path.join(base, d)
    return None


# ==========================================================
# WORKER
# ==========================================================

class WorkerSignals(QObject):
    finished = pyqtSignal(object)

class UARTWorker(QRunnable):
    def __init__(self, freq, water_ref):
        super().__init__()
        self.freq = freq
        self.water_ref = water_ref
        self.signals = WorkerSignals()

    def run(self):
        try:
            V_S = process_usart_data("/dev/serial0", Vref=976e-6)

            if V_S is None or len(V_S) == 0:
                self.signals.finished.emit(None)
                return

            
            V_W = self.water_ref.get_signal(self.freq)
            n = min(len(V_S), len(V_W))
            V_S = V_S[:n].astype(np.float32, copy=False)
            V_W = V_W[:n].astype(np.float32, copy=False)

            att, windowed = compute_attenuation(V_S, V_W, 50e6, self.freq * 1e6, 0.02)

            self.signals.finished.emit((att, windowed))

        except Exception as e:
            print("UART error:", e)
            self.signals.finished.emit(None)


# ==========================================================
# MAIN UI
# ==========================================================

class AttenuationUI(QWidget):
    up_signal = pyqtSignal()
    down_signal = pyqtSignal()
    enter_signal = pyqtSignal()
    export_signal = pyqtSignal()
    shutdown_signal = pyqtSignal()
    freq_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        import pyqtgraph as pg
        self.pg = pg

        self.up_signal.connect(self.on_up)
        self.down_signal.connect(self.on_down)
        self.enter_signal.connect(self.on_enter)
        self.export_signal.connect(self.export_selected)
        self.shutdown_signal.connect(self.shutdown)
        self.freq_signal.connect(self.adjust_frequency)
        self.setWindowTitle("Attenuation System")
        self.showFullScreen()

        self.mode = "browser"
        self.session = None
        self.water_ref = WaterReference("WaterReferences.csv")
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(2)
        self.current_freq = 1.0

        # ================= MAIN LAYOUT =================
        main_layout = QHBoxLayout(self)

        # LEFT FILE LIST
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_file)

        self.file_list.setMaximumWidth(300)
        main_layout.addWidget(self.file_list)

        self.refresh_files()

        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)

        # RIGHT SIDE
        right = QVBoxLayout()

        # PLOT
        self.plot = self.pg.PlotWidget()
        self.plot.setLabel("left", "Attenuation (dB)")
        self.plot.setLabel("bottom", "Frequency (MHz)")
        right.addWidget(self.plot)
        self.scatter = self.plot.plot([], [], pen=None, symbol='o')
        self.line = self.plot.plot([], [])
        # ================= TERMINAL PANEL =================
        terminal_frame = QFrame()
        terminal_frame.setStyleSheet("""
            QFrame {
                background-color: #111111;
                border: 2px solid #333;
                border-radius: 8px;
            }
        """)

        terminal_layout = QVBoxLayout(terminal_frame)

        self.label = QLabel("Ready")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # 🔥 TERMINAL STYLE TEXT
        self.label.setStyleSheet("""
            QLabel {
                color: #d4d4d4;
                font-family: Consolas, "Courier New", monospace;
                font-size: 20px;
                padding: 15px;
                background-color: #1e1e1e;
            }
        """)
        terminal_layout.addWidget(self.label)

        # make it big (bottom panel feel)
        terminal_frame.setMinimumHeight(220)

        right.addWidget(terminal_frame)

        main_layout.addLayout(right)
        self.setLayout(main_layout)

        # GPIO INIT
        if running_on_pi():
            from gpiozero import Button

            global GPIO_UP, GPIO_DOWN, GPIO_ENTER
            global GPIO_EXPORT, GPIO_SHUTDOWN, GPIO_FREQ

            debounce = 0.2

            GPIO_UP = Button(5, bounce_time=debounce)
            GPIO_DOWN = Button(6, bounce_time=debounce)
            GPIO_ENTER = Button(13, bounce_time=debounce)
            GPIO_EXPORT = Button(19, bounce_time=debounce)
            GPIO_SHUTDOWN = Button(21, bounce_time=debounce)
            GPIO_FREQ = Button(26, bounce_time=debounce)

            GPIO_ENTER.when_pressed = self.enter_signal.emit
            GPIO_UP.when_pressed = self.up_signal.emit
            GPIO_DOWN.when_pressed = self.down_signal.emit
            GPIO_EXPORT.when_pressed = self.export_signal.emit
            GPIO_SHUTDOWN.when_pressed = self.shutdown_signal.emit
            GPIO_FREQ.when_pressed = self.freq_signal.emit

        self.update_label()

    # ---------------- FILE HANDLING ----------------
    def refresh_files(self):
        self.file_list.clear()
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv"):
                self.file_list.addItem(f)

    def load_file(self, current):
        if not current:
            return

        path = os.path.join(DATA_DIR, current.text())

        try:
            self.session = SweepSession.load(path, self.water_ref)
        except Exception as e:
            self.update_label(f"> Load failed: {e}")
            return

        if self.session is None:
            return

        self.mode = "browser"
        self.update_plot()
        self.update_label(f"> Loaded: {current.text()}")
    # ---------------- PLOT ----------------
    def update_plot(self):
        if self.session is None:
            return
        x = np.asarray(self.session.frequencies, dtype=np.float32)
        y = np.asarray(self.session.attenuations, dtype=np.float32)
        self.scatter.setData(x, y)

        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xline = np.linspace(min(x), max(x), 100)
            yline = m * xline + b
            self.line.setData(xline, yline)
    # ---------------- LABEL ----------------
    def update_label(self, message=None):
        if message:
            self.label.setText(message)
            return

        if self.mode == "browser":
            current = self.file_list.currentItem()
            self.label.setText(
                f"> Browser Mode\n"
                f"> Selected: {current.text() if current else 'None'}\n\n"
                f"> Press FREQ to start new sweep\n"
                f"> Press ENTER to view data"
            )

        elif self.mode == "sampling":
            self.label.setText(
                f"> Sampling Mode\n\n"
                f"> Frequency: {self.current_freq:.2f} MHz\n\n"
                f"> UP/DOWN : Adjust\n"
                f"> ENTER   : Measure"
            )

        elif self.mode == "detail":
            text = "> DETAIL VIEW\n\n"
            for i, (f, a) in enumerate(zip(self.session.frequencies, self.session.attenuations)):
                text += f"> {i+1}) {f} MHz → {a:.3f} dB\n"

            if self.session.coefficient:
                text += f"\n> Coeff: {self.session.coefficient:.5f}"

            text += "\n\n> UP/DOWN to return"
            self.label.setText(text)


        elif self.mode == "post_decision":
            self.label.setText(
                f"> Sweep Step Complete\n\n"
                f"> ENTER : Save sweep\n"
                f"> FREQ  : New sample\n"
                f"> UP/DOWN : Cancel"
            )

    # ---------------- SAMPLING ----------------
    def adjust_frequency(self):
        STEP = 0.05
        MIN_F = 0.8
        MAX_F = 1.2

        if self.mode == "browser":
            self.session = SweepSession(self.water_ref)
            self.mode = "sampling"
            self.current_freq = MIN_F

        elif self.mode == "sampling":
            self.current_freq = round(min(MAX_F, self.current_freq + STEP), 2)

        elif self.mode == "post_decision":
            # keep same freq but return to sampling
            self.mode = "sampling"
            self.current_freq = round(self.current_freq, 2)

        self.update_label()

    def on_enter(self):
        if self.mode == "sampling":
            self.update_label(f"> Waiting for UART @ {self.current_freq:.2f} MHz...")

            worker = UARTWorker(self.current_freq, self.water_ref)
            worker.signals.finished.connect(self.handle_uart_result)
            self.threadpool.start(worker)

        elif self.mode == "post_decision":
            # SAVE FINALIZED SWEEP
            try:
                self.session.finalize()
                self.session.save()

                self.refresh_files()
                self.mode = "browser"
                self.update_label("> Sweep saved successfully")
                self.update_plot()

            except Exception as e:
                self.update_label(f"> Save error: {str(e)}")

        elif self.mode == "browser":
            current = self.file_list.currentItem()
            if current:
                self.load_file(current)
                self.mode = "detail"
                self.update_label()
    # ---------------- NAV ----------------
    def on_up(self):
        STEP = 0.05
        MAX_F = 1.2

        if self.mode == "sampling":
            self.current_freq = round(min(MAX_F, self.current_freq + STEP), 2)

        elif self.mode == "post_decision":
            self.mode = "browser"

        elif self.mode == "browser":
            row = self.file_list.currentRow()
            if row > 0:
                self.file_list.setCurrentRow(row - 1)
        elif self.mode == "detail":
            self.mode = "browser"

        self.update_label()

    def on_down(self):
        STEP = 0.05
        MIN_F = 0.8

        if self.mode == "sampling":
            self.current_freq = round(max(MIN_F, self.current_freq - STEP), 2)

        elif self.mode == "post_decision":
            self.mode = "browser"

        elif self.mode == "browser":
            row = self.file_list.currentRow()
            if row < self.file_list.count() - 1:
                self.file_list.setCurrentRow(row + 1)
                
        elif self.mode == "detail":
            self.mode = "browser"

        self.update_label()

    # ---------------- EXPORT ----------------
    def export_selected(self):
        if not self.session:
            self.update_label("> No sweep selected")
            return

        usb = find_usb()
        if not usb:
            self.update_label("> No USB detected")
            return

        path = os.path.join(usb, f"{self.session.sweep_id}.csv")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["coefficient", self.session.coefficient])
            writer.writerow([])
            writer.writerow(["frequency", "attenuation"])

            for fval, aval in zip(self.session.frequencies, self.session.attenuations):
                writer.writerow([fval, aval])

        self.update_label(f"> Exported to {usb}")

    # ---------------- SHUTDOWN ----------------
    def shutdown(self):
        self.update_label("> Shutting down...")
        if running_on_pi():
            subprocess.run(["sudo", "shutdown", "-h", "now"])

    def handle_uart_result(self, result):
        if result is None:
            self.update_label("> Error reading UART")
            return

        if self.session is None:
            self.session = SweepSession(self.water_ref)

        att, windowed = result

        self.session.frequencies.append(float(self.current_freq))
        self.session.attenuations.append(float(att))
        self.session.windowed_signals.append(windowed)

        self.update_plot()
        self.mode = "post_decision"
        self.update_label()
    # ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AttenuationUI()
    w.show()
    sys.exit(app.exec_())