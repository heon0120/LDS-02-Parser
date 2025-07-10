import serial
import struct
import math
import time
import threading
import queue
import json
import csv
from collections import deque
from typing import Optional, Dict, List, Callable, Any
import logging

# 상수 정의
HEADER_BYTE = 0x54
LENGTH_BYTE = 0x2C
PACKET_SIZE = 47
MAX_DISTANCE = 6000
DEFAULT_BAUDRATE = 115200


class LDS02Point:
    """LDS-02 포인트 데이터를 나타내는 클래스"""

    def __init__(self, angle_deg: float, distance_mm: int, confidence: int):
        self.angle_deg = angle_deg
        self.distance_mm = distance_mm
        self.confidence = confidence

        # 직교 좌표계 변환
        self.angle_rad = math.radians(angle_deg)
        self.x_mm = distance_mm * math.sin(self.angle_rad)
        self.y_mm = distance_mm * math.cos(self.angle_rad)

    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환"""
        return {
            'angle_deg': self.angle_deg,
            'angle_rad': self.angle_rad,
            'distance_mm': self.distance_mm,
            'confidence': self.confidence,
            'x_mm': self.x_mm,
            'y_mm': self.y_mm
        }

    def __repr__(self):
        return f"LDS02Point(angle={self.angle_deg:.1f}°, dist={self.distance_mm}mm, conf={self.confidence})"


class LDS02Frame:
    """LDS-02 프레임 데이터를 나타내는 클래스"""

    def __init__(self, speed: float, timestamp: int, points: List[LDS02Point]):
        self.speed = speed  # RPM
        self.timestamp = timestamp
        self.points = points
        self.frame_time = time.time()

    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환"""
        return {
            'speed': self.speed,
            'timestamp': self.timestamp,
            'frame_time': self.frame_time,
            'points': [point.to_dict() for point in self.points]
        }

    def __repr__(self):
        return f"LDS02Frame(speed={self.speed:.1f}RPM, points={len(self.points)})"


class LDS02Parser:
    """LDS-02 LiDAR 센서 범용 파서 클래스"""

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE,
                 max_distance: int = MAX_DISTANCE,
                 data_callback: Optional[Callable[[LDS02Frame], None]] = None,
                 error_callback: Optional[Callable[[str], None]] = None):
        """
        Args:
            port: 시리얼 포트 (예: 'COM3', '/dev/ttyUSB0')
            baudrate: 보드레이트 (기본값: 115200)
            max_distance: 최대 거리 필터링 (mm)
            data_callback: 데이터 수신 시 호출될 콜백 함수
            error_callback: 오류 발생 시 호출될 콜백 함수
        """
        self.port = port
        self.baudrate = baudrate
        self.max_distance = max_distance
        self.data_callback = data_callback
        self.error_callback = error_callback

        # 시리얼 통신
        self.serial_port = None
        self.buffer = bytearray()

        # 스레드 관리
        self.running = False
        self.read_thread = None

        # 통계 정보
        self.packet_count = 0
        self.error_count = 0
        self.latest_speed = 0.0
        self.start_time = time.time()

        # 데이터 큐
        self.frame_queue = queue.Queue(maxsize=200)

        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def connect(self) -> bool:
        """시리얼 포트 연결"""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            self.serial_port.flushInput()
            self.logger.info(f"Connected to {self.port} at {self.baudrate} bps")
            return True
        except serial.SerialException as e:
            error_msg = f"Serial connection failed: {e}"
            self.logger.error(error_msg)
            if self.error_callback:
                self.error_callback(error_msg)
            return False

    def disconnect(self):
        """시리얼 포트 연결 해제"""
        self.stop()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.logger.info("Serial port closed")

    def start(self):
        """데이터 읽기 시작"""
        if not self.running and self.serial_port and self.serial_port.is_open:
            self.running = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            self.logger.info("Data reading started")

    def stop(self):
        """데이터 읽기 중지"""
        if self.running:
            self.running = False
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=1)
            self.logger.info("Data reading stopped")

    def _read_loop(self):
        """데이터 읽기 루프 (별도 스레드에서 실행)"""
        while self.running:
            try:
                # 시리얼 데이터 읽기
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    self.buffer.extend(data)

                # 패킷 파싱
                while len(self.buffer) >= PACKET_SIZE:
                    # 헤더 검증
                    if self.buffer[0] != HEADER_BYTE or self.buffer[1] != LENGTH_BYTE:
                        del self.buffer[0]
                        self.error_count += 1
                        continue

                    # 패킷 추출
                    packet = self.buffer[:PACKET_SIZE]
                    del self.buffer[:PACKET_SIZE]

                    # 패킷 파싱
                    frame = self._parse_packet(packet)
                    if frame:
                        self.packet_count += 1
                        self.latest_speed = frame.speed

                        # 콜백 호출
                        if self.data_callback:
                            self.data_callback(frame)

                        # 큐에 추가
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    else:
                        self.error_count += 1

                time.sleep(0.001)  # CPU 사용량 조절

            except serial.SerialException as e:
                error_msg = f"Serial port error: {e}"
                self.logger.error(error_msg)
                if self.error_callback:
                    self.error_callback(error_msg)
                self.running = False
            except Exception as e:
                error_msg = f"Read loop error: {e}"
                self.logger.error(error_msg)
                if self.error_callback:
                    self.error_callback(error_msg)
                self.buffer.clear()
                time.sleep(0.01)

    def _parse_packet(self, packet: bytes) -> Optional[LDS02Frame]:
        """패킷을 파싱하여 LDS02Frame 객체로 변환"""
        try:
            if len(packet) != PACKET_SIZE:
                return None

            # 데이터 영역 추출 (헤더 제외)
            data = packet[2:46]

            # 기본 정보 추출
            speed = struct.unpack('<H', data[0:2])[0] / 100.0  # RPM
            start_angle = struct.unpack('<H', data[2:4])[0] / 100.0  # 시작 각도
            end_angle = struct.unpack('<H', data[40:42])[0] / 100.0  # 끝 각도
            timestamp = struct.unpack('<H', data[42:44])[0]  # 타임스탬프

            # 각도 차이 계산
            angle_diff = (end_angle - start_angle + 360) % 360

            # 포인트 데이터 추출
            points = []
            for i in range(12):  # 12개의 포인트
                offset = 4 + i * 3
                distance = struct.unpack('<H', data[offset:offset + 2])[0]
                confidence = data[offset + 2]

                # 유효한 거리 값만 처리
                if 0 < distance <= self.max_distance:
                    angle = (start_angle + angle_diff * (i / 12.0)) % 360
                    point = LDS02Point(angle, distance, confidence)
                    points.append(point)

            return LDS02Frame(speed, timestamp, points)

        except (struct.error, IndexError) as e:
            self.logger.debug(f"Packet parsing error: {e}")
            return None

    def get_frame(self, timeout: float = 1.0) -> Optional[LDS02Frame]:
        """큐에서 프레임 가져오기"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        runtime = time.time() - self.start_time
        return {
            'packets': self.packet_count,
            'errors': self.error_count,
            'error_rate': self.error_count / max(1, self.packet_count + self.error_count),
            'latest_speed': self.latest_speed,
            'queue_size': self.frame_queue.qsize(),
            'runtime': runtime,
            'packet_rate': self.packet_count / max(1, runtime)
        }

    def save_to_csv(self, filename: str, max_frames: int = 1000):
        """CSV 파일로 저장"""
        frames = []
        frame_count = 0

        # 큐에서 프레임 수집
        while not self.frame_queue.empty() and frame_count < max_frames:
            frame = self.frame_queue.get_nowait()
            frames.append(frame)
            frame_count += 1

        if not frames:
            self.logger.warning("No data to save")
            return

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['frame_time', 'speed', 'timestamp', 'angle_deg',
                              'distance_mm', 'confidence', 'x_mm', 'y_mm']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for frame in frames:
                    for point in frame.points:
                        writer.writerow({
                            'frame_time': frame.frame_time,
                            'speed': frame.speed,
                            'timestamp': frame.timestamp,
                            'angle_deg': point.angle_deg,
                            'distance_mm': point.distance_mm,
                            'confidence': point.confidence,
                            'x_mm': point.x_mm,
                            'y_mm': point.y_mm
                        })

            self.logger.info(f"Data saved to {filename} ({len(frames)} frames)")

        except Exception as e:
            error_msg = f"Failed to save data: {e}"
            self.logger.error(error_msg)
            if self.error_callback:
                self.error_callback(error_msg)

    def save_to_json(self, filename: str, max_frames: int = 1000):
        """JSON 파일로 저장"""
        frames = []
        frame_count = 0

        # 큐에서 프레임 수집
        while not self.frame_queue.empty() and frame_count < max_frames:
            frame = self.frame_queue.get_nowait()
            frames.append(frame.to_dict())
            frame_count += 1

        if not frames:
            self.logger.warning("No data to save")
            return

        try:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump({
                    'metadata': {
                        'port': self.port,
                        'baudrate': self.baudrate,
                        'max_distance': self.max_distance,
                        'stats': self.get_stats()
                    },
                    'frames': frames
                }, jsonfile, indent=2)

            self.logger.info(f"Data saved to {filename} ({len(frames)} frames)")

        except Exception as e:
            error_msg = f"Failed to save data: {e}"
            self.logger.error(error_msg)
            if self.error_callback:
                self.error_callback(error_msg)


