import serial
import struct
import math
import time
import numpy as np
from collections import deque
import threading
import queue
import warnings
import argparse
import traceback
import sys
import csv

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QSlider, \
    QSpinBox, QGridLayout
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

warnings.filterwarnings("ignore", category=UserWarning)

HEADER_BYTE = 0x54
LENGTH_BYTE = 0x2C
PACKET_SIZE = 47

MAX_DISTANCE = 6000
MAX_POINTS = 360
UPDATE_INTERVAL = 16


def parse_args():
    parser = argparse.ArgumentParser(description="LDS-02 Lidar Visualizer")
    parser.add_argument('--port', type=str, default='COM15', help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    return parser.parse_args()


class Lds02Parser:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buffer = bytearray()
        self.data_queue = queue.Queue(maxsize=200)
        self.running = False
        self.thread = None
        self.packet_count = 0
        self.error_count = 0
        self.latest_speed = 0.0

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.ser.flushInput()
            print(f"[✔] Connected to {self.port} at {self.baudrate} bps")
            return True
        except serial.SerialException as e:
            print(f"[!] Serial connection failed: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[✔] Serial port closed")

    def start_reading(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.read_loop, daemon=True)
            self.thread.start()

    def stop_reading(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    def read_loop(self):
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    self.buffer.extend(self.ser.read(self.ser.in_waiting))

                while len(self.buffer) >= PACKET_SIZE:
                    if self.buffer[0] != HEADER_BYTE or self.buffer[1] != LENGTH_BYTE:
                        del self.buffer[0]
                        self.error_count += 1
                        continue

                    if len(self.buffer) < PACKET_SIZE:
                        break

                    packet = self.buffer[:PACKET_SIZE]
                    del self.buffer[:PACKET_SIZE]

                    parsed = self.parse_packet(packet)
                    if parsed:
                        self.packet_count += 1
                        self.latest_speed = parsed['speed']
                        if not self.data_queue.full():
                            self.data_queue.put(parsed)
                    else:
                        self.error_count += 1

                time.sleep(0.001)
            except serial.SerialException as e:
                print(f"[!] Serial port error: {e}")
                self.running = False
            except Exception as e:
                print(f"[!] Read loop error: {e}")
                traceback.print_exc()
                self.buffer.clear()
                time.sleep(0.01)

    def parse_packet(self, packet):
        try:
            if len(packet) != PACKET_SIZE:
                return None
            data = packet[2:46]

            speed = struct.unpack('<H', data[0:2])[0] / 100.0
            start_angle = struct.unpack('<H', data[2:4])[0] / 100.0
            end_angle = struct.unpack('<H', data[40:42])[0] / 100.0
            timestamp = struct.unpack('<H', data[42:44])[0]

            angle_diff = (end_angle - start_angle + 360) % 360

            points = []
            for i in range(12):
                offset = 4 + i * 3
                dist = struct.unpack('<H', data[offset:offset + 2])[0]
                conf = data[offset + 2]

                if 0 < dist < MAX_DISTANCE:
                    angle = (start_angle + angle_diff * (i / 12.0)) % 360
                    points.append({'angle_deg': angle, 'distance_mm': dist, 'confidence': conf})

            return {'speed': speed, 'timestamp': timestamp, 'points': points}
        except struct.error as e:
            return None
        except Exception as e:
            return None

    def get_stats(self):
        return {
            'packets': self.packet_count,
            'errors': self.error_count,
            'queue_size': self.data_queue.qsize(),
            'latest_speed': self.latest_speed
        }


class RealTimeLidarVisualizer(QMainWindow):
    connection_status_signal = pyqtSignal(bool)

    def __init__(self, parser):
        super().__init__()
        self.parser = parser

        self.lidar_points = deque(maxlen=MAX_POINTS * 10)

        self.distance_history = deque(maxlen=5000)
        self.speed_history = deque(maxlen=500)
        self.time_history = deque(maxlen=500)

        self.fps_counter = deque(maxlen=60)
        self.last_update_time = time.time()
        self.start_time = time.time()

        self.is_paused = False
        self.current_max_distance = MAX_DISTANCE

        self.setup_ui()
        self.setup_timer()

        self.connection_status_signal.connect(self.update_connection_status)

    def setup_ui(self):
        self.setWindowTitle("LDS-02 Lidar Visualizer")
        self.setGeometry(100, 100, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_bar_layout = QHBoxLayout()

        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("padding: 5px; border-radius: 5px; background-color: #333; color: white;")
        top_bar_layout.addWidget(self.status_label, 3)

        self.rpm_label = QLabel("RPM: 0.0")
        self.rpm_label.setAlignment(Qt.AlignCenter)
        self.rpm_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.rpm_label.setStyleSheet(
            "padding: 5px; border: 1px solid #555; border-radius: 5px; background-color: #222; color: #00FFFF;")
        top_bar_layout.addWidget(self.rpm_label, 1)

        button_layout = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)

        self.clear_button = QPushButton("Clear Data")
        self.clear_button.clicked.connect(self.clear_data)
        button_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self.save_data)
        button_layout.addWidget(self.save_button)

        max_dist_label = QLabel("Max Dist (mm):")
        max_dist_label.setFont(QFont("Arial", 10))
        button_layout.addWidget(max_dist_label)
        self.max_dist_spinbox = QSpinBox()
        self.max_dist_spinbox.setRange(1000, 10000)
        self.max_dist_spinbox.setSingleStep(500)
        self.max_dist_spinbox.setValue(MAX_DISTANCE)
        self.max_dist_spinbox.valueChanged.connect(self.update_max_distance)
        button_layout.addWidget(self.max_dist_spinbox)

        top_bar_layout.addLayout(button_layout, 2)

        main_layout.addLayout(top_bar_layout)

        graph_grid_layout = QGridLayout()
        main_layout.addLayout(graph_grid_layout)

        self.xy_widget = pg.PlotWidget(title="XY View (mm)")
        self.xy_widget.setLabel('left', 'Y (mm)')
        self.xy_widget.setLabel('bottom', 'X (mm)')
        self.xy_widget.setXRange(-self.current_max_distance, self.current_max_distance)
        self.xy_widget.setYRange(-self.current_max_distance, self.current_max_distance)
        self.xy_widget.setAspectLocked(True)
        self.xy_widget.showGrid(True, True)
        self.xy_widget.setBackground('black')
        self.add_circular_grid()
        self.scatter_plot = pg.ScatterPlotItem()
        self.xy_widget.addItem(self.scatter_plot)
        graph_grid_layout.addWidget(self.xy_widget, 0, 0, 2, 2)

        self.angle_dist_widget = pg.PlotWidget(title="Angle vs. Distance")
        self.angle_dist_widget.setLabel('left', 'Distance (mm)')
        self.angle_dist_widget.setLabel('bottom', 'Angle (degrees)')
        self.angle_dist_widget.setXRange(0, 360)
        self.angle_dist_widget.setYRange(0, self.current_max_distance)
        self.angle_dist_widget.showGrid(True, True)
        self.angle_dist_widget.setBackground('black')
        self.angle_dist_plot = pg.PlotCurveItem(pen='yellow')
        self.angle_dist_widget.addItem(self.angle_dist_plot)
        graph_grid_layout.addWidget(self.angle_dist_widget, 0, 2)

        self.conf_angle_widget = pg.PlotWidget(title="Confidence vs. Angle")
        self.conf_angle_widget.setLabel('left', 'Confidence (0-255)')
        self.conf_angle_widget.setLabel('bottom', 'Angle (degrees)')
        self.conf_angle_widget.setXRange(0, 360)
        self.conf_angle_widget.setYRange(0, 255)
        self.conf_angle_widget.showGrid(True, True)
        self.conf_angle_widget.setBackground('black')
        self.conf_angle_plot = pg.PlotCurveItem(pen='magenta')
        self.conf_angle_widget.addItem(self.conf_angle_plot)
        graph_grid_layout.addWidget(self.conf_angle_widget, 1, 2)

        self.hist_widget = pg.PlotWidget(title="Distance Histogram")
        self.hist_widget.setLabel('left', 'Count')
        self.hist_widget.setLabel('bottom', 'Distance (mm)')
        self.hist_widget.setXRange(0, self.current_max_distance)
        self.hist_widget.setBackground('black')
        self.hist_widget.showGrid(True, True)
        self.hist_plot = pg.BarGraphItem(x=[], height=[], width=0.8, brush=(0, 255, 255, 80))
        self.hist_widget.addItem(self.hist_plot)
        graph_grid_layout.addWidget(self.hist_widget, 2, 0)

        self.rpm_trend_widget = pg.PlotWidget(title="RPM Trend")
        self.rpm_trend_widget.setLabel('left', 'RPM')
        self.rpm_trend_widget.setLabel('bottom', 'Time (s)')
        self.rpm_trend_widget.setYRange(0, 600)
        self.rpm_trend_widget.showGrid(True, True)
        self.rpm_trend_widget.setBackground('black')
        self.rpm_trend_plot = pg.PlotCurveItem(pen='lime')
        self.rpm_trend_widget.addItem(self.rpm_trend_plot)
        graph_grid_layout.addWidget(self.rpm_trend_widget, 2, 1, 1, 2)

    def add_circular_grid(self):
        for item in self.xy_widget.items():
            if isinstance(item, pg.ROI) or isinstance(item, pg.graphicsItems.InfiniteLine.InfiniteLine):
                self.xy_widget.removeItem(item)

        for radius in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
            if radius <= self.current_max_distance:
                circle = pg.CircleROI([0, 0], [radius * 2, radius * 2],
                                      movable=False, pen=pg.mkPen('gray', width=1, dash=[2, 2]))
                circle.removeHandle(0)
                self.xy_widget.addItem(circle)

        for angle in range(0, 360, 45):
            line = pg.InfiniteLine(angle=angle, pen=pg.mkPen('gray', width=1, dash=[2, 2]))
            self.xy_widget.addItem(line)

    def update_max_distance(self, value):
        self.current_max_distance = value
        self.xy_widget.setXRange(-self.current_max_distance, self.current_max_distance)
        self.xy_widget.setYRange(-self.current_max_distance, self.current_max_distance)
        self.angle_dist_widget.setYRange(0, self.current_max_distance)
        self.hist_widget.setXRange(0, self.current_max_distance)
        self.add_circular_grid()

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(UPDATE_INTERVAL)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.timer.stop()
            self.pause_button.setText("Resume")
            self.status_label.setText("Paused | " + self.status_label.text())
            self.status_label.setStyleSheet("padding: 5px; border-radius: 5px; background-color: orange; color: white;")
        else:
            self.timer.start(UPDATE_INTERVAL)
            self.pause_button.setText("Pause")
            self.status_label.setStyleSheet("padding: 5px; border-radius: 5px; background-color: #333; color: white;")
            self.last_update_time = time.time()

    def clear_data(self):
        self.lidar_points.clear()
        self.distance_history.clear()
        self.speed_history.clear()
        self.time_history.clear()
        self.fps_counter.clear()

        self.scatter_plot.setData([], [])
        self.angle_dist_plot.setData([], [])
        self.conf_angle_plot.setData([], [])
        self.hist_plot.setOpts(x=[], height=[])
        self.rpm_trend_plot.setData([], [])

        self.parser.packet_count = 0
        self.parser.error_count = 0
        self.parser.latest_speed = 0.0
        while not self.parser.data_queue.empty():
            self.parser.data_queue.get_nowait()

        self.status_label.setText("Data Cleared. Waiting for new data...")
        self.rpm_label.setText("RPM: 0.0")
        self.last_update_time = time.time()
        self.start_time = time.time()

    def save_data(self):
        if not self.lidar_points:
            print("[!] No data to save.")
            return

        filename = f"lidar_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['X_mm', 'Y_mm', 'Angle_deg', 'Distance_mm', 'Confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for point in self.lidar_points:
                    writer.writerow({
                        'X_mm': point['x'],
                        'Y_mm': point['y'],
                        'Angle_deg': math.degrees(point['angle_rad']),
                        'Distance_mm': point['dist'],
                        'Confidence': point['conf']
                    })
            print(f"[✔] Data saved to {filename}")
            self.status_label.setText(f"Data saved to {filename} | FPS: {np.mean(self.fps_counter):.1f}")
        except Exception as e:
            print(f"[X] Failed to save data: {e}")
            self.status_label.setText(f"Error saving data: {e}")

    def update_connection_status(self, connected):
        if connected:
            self.status_label.setStyleSheet(
                "padding: 5px; border-radius: 5px; background-color: #008000; color: white;")
        else:
            self.status_label.setStyleSheet(
                "padding: 5px; border-radius: 5px; background-color: #FF0000; color: white;")

    def update_plots(self):
        if self.is_paused:
            return

        now = time.time()
        new_data_count = 0

        while not self.parser.data_queue.empty() and new_data_count < 50:
            data = self.parser.data_queue.get_nowait()
            self.speed_history.append(data['speed'])
            self.time_history.append(now - self.start_time)

            for pt in data['points']:
                rad = math.radians(pt['angle_deg'])
                dist = pt['distance_mm']
                conf = pt['confidence']
                x = dist * math.sin(rad)
                y = dist * math.cos(rad)

                self.lidar_points.append({'angle_rad': rad, 'dist': dist, 'conf': conf, 'x': x, 'y': y})
                self.distance_history.append(dist)

            new_data_count += 1

        points_to_render = list(self.lidar_points)[-MAX_POINTS:]

        if points_to_render:
            all_x_coords = np.array([p['x'] for p in points_to_render])
            all_y_coords = np.array([p['y'] for p in points_to_render])
            all_angles_rad = np.array([p['angle_rad'] for p in points_to_render])
            all_distances = np.array([p['dist'] for p in points_to_render])
            all_confidences = np.array([p['conf'] for p in points_to_render])
        else:
            all_x_coords, all_y_coords, all_angles_rad, all_distances, all_confidences = \
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        if all_x_coords.size > 0:
            colors = []
            for conf in all_confidences:
                if conf < 50:
                    colors.append((255, 68, 68, 200))
                elif conf < 100:
                    colors.append((255, 255, 68, 200))
                else:
                    colors.append((68, 255, 68, 200))

            valid_indices = (np.sqrt(all_x_coords ** 2 + all_y_coords ** 2) <= self.current_max_distance)
            self.scatter_plot.setData(
                x=all_x_coords[valid_indices],
                y=all_y_coords[valid_indices],
                brush=(255, 255, 255, 120),
                size=5,
                pen=None
            )
        else:
            self.scatter_plot.setData([], [])

        if all_angles_rad.size > 0 and all_distances.size > 0:
            angles_deg = np.degrees(all_angles_rad) % 360

            sort_indices = np.argsort(angles_deg)
            self.angle_dist_plot.setData(angles_deg[sort_indices], all_distances[sort_indices])
        else:
            self.angle_dist_plot.setData([], [])

        if all_angles_rad.size > 0 and all_confidences.size > 0:
            angles_deg_conf = np.degrees(all_angles_rad) % 360
            sort_indices_conf = np.argsort(angles_deg_conf)
            self.conf_angle_plot.setData(angles_deg_conf[sort_indices_conf], all_confidences[sort_indices_conf])
        else:
            self.conf_angle_plot.setData([], [])

        if self.distance_history:
            hist, bins = np.histogram(self.distance_history, bins=50, range=(0, self.current_max_distance))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            self.hist_plot.setOpts(x=bin_centers, height=hist)
        else:
            self.hist_plot.setOpts(x=[], height=[])

        if self.time_history and self.speed_history:
            self.rpm_trend_plot.setData(np.array(self.time_history), np.array(self.speed_history))
            if self.time_history:
                max_time = self.time_history[-1]
                min_time = max(0, max_time - 10)
                self.rpm_trend_widget.setXRange(min_time, max_time + 0.5)
        else:
            self.rpm_trend_plot.setData([], [])

        fps = 1 / (now - self.last_update_time + 1e-6)
        self.fps_counter.append(fps)
        avg_fps = np.mean(self.fps_counter)

        stats = self.parser.get_stats()
        status_text = (f"FPS: {avg_fps:.1f} | Packets: {stats['packets']} | Errors: {stats['errors']} | "
                       f"Queue: {stats['queue_size']}")
        self.status_label.setText(status_text)
        self.rpm_label.setText(f"RPM: {stats['latest_speed']:.1f}")
        self.last_update_time = now

    def closeEvent(self, event):
        print("[!] Closing application...")
        self.timer.stop()
        self.parser.stop_reading()
        self.parser.disconnect()
        event.accept()


def main():
    args = parse_args()

    app = QApplication(sys.argv)

    parser = Lds02Parser(args.port, args.baud)

    visualizer = RealTimeLidarVisualizer(parser)
    visualizer.show()

    if parser.connect():
        visualizer.connection_status_signal.emit(True)
        parser.start_reading()
        print("[✔] Visualization started")
    else:
        visualizer.connection_status_signal.emit(False)
        print("[X] Failed to connect. Check the COM port and device.")

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    except Exception as e:
        print(f"[X] An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        parser.stop_reading()
        parser.disconnect()
        print("[✔] Application terminated.")


if __name__ == "__main__":
    main()
