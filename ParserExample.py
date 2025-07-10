import numpy
import matplotlib.pyplot as plt
from LDS02Parse import LDS02Parser, LDS02Frame
import time
import queue

# 실시간 플로팅을 위한 설정
plt.ion() # 인터랙티브 모드 켜기
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
scatter_plot = None # 초기 산점도 객체

def plot_lidar_data_realtime(frame: LDS02Frame):
    # 실시간 LiDAR 데이터 시각화 (업데이트)
    global scatter_plot

    if not frame.points:
        return

    angles = [p.angle_rad for p in frame.points]
    distances = [p.distance_mm for p in frame.points]

    if scatter_plot is None:
        # 최초 플로팅
        scatter_plot = ax.scatter(angles, distances, s=2, alpha=0.6)
        ax.set_title(f'LiDAR Data - Speed: {frame.speed:.1f} RPM')
        ax.set_ylim(0, 6000) # 거리 범위 설정 (필요에 따라 조절)
    else:
        # 데이터 업데이트
        offsets = numpy.array([angles, distances]).T
        scatter_plot.set_offsets(offsets)
        ax.set_title(f'LiDAR Data - Speed: {frame.speed:.1f} RPM')

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def data_received_callback(frame: LDS02Frame):
    # UI 스레드에서 플로팅이 이루어지도록 큐에 프레임 추가
    data_queue.put(frame)

def error_occurred_callback(error_msg: str):
    print(f"오류: {error_msg}")


if __name__ == "__main__":
    # LiDAR 데이터를 받을 큐 생성
    data_queue = queue.Queue(maxsize=10)

    # 시리얼 포트 설정 (실제 포트에 맞게 변경해주세요)
    # 예: Windows의 경우 'COMx', Linux의 경우 '/dev/ttyUSBx'
    # 현재 환경에 맞는 시리얼 포트를 검색하여 설정해야 합니다.
    # 포트 이름은 장치 관리자(Windows) 또는 dmesg | grep tty (Linux) 등으로 확인할 수 있습니다.
    lidar_port = 'COM7' # 실제 사용 중인 시리얼 포트로 변경하세요.

    # LDS02Parser 인스턴스 생성
    # data_callback 함수로 data_received_callback을 지정하여 데이터 수신 시 큐에 넣도록 합니다.
    parser = LDS02Parser(
        port=lidar_port,
        baudrate=115200,
        data_callback=data_received_callback,
        error_callback=error_occurred_callback
    )

    if parser.connect():
        parser.start()
        print("LiDAR 데이터 수신 시작. 플로팅 창을 확인하세요.")
        print("종료하려면 Ctrl+C를 누르세요.")

        try:
            while True:
                # 큐에서 프레임을 가져와서 플로팅
                try:
                    frame_to_plot = data_queue.get(timeout=0.1) # 큐에서 데이터 가져오기 (비블로킹)
                    plot_lidar_data_realtime(frame_to_plot)
                except queue.Empty:
                    pass # 큐가 비어있으면 계속 진행

                # 주기적으로 통계 출력 (선택 사항)
                # print(parser.get_stats())
                time.sleep(0.01) # CPU 사용량 줄이기

        except KeyboardInterrupt:
            print("\n프로그램 종료 요청.")
        finally:
            parser.stop()
            parser.disconnect()
            plt.ioff() # 인터랙티브 모드 끄기
            plt.close(fig) # 플로팅 창 닫기
            print("LiDAR 파서 및 플로팅 종료.")
    else:
        print(f"'{lidar_port}' 포트에 연결할 수 없습니다. 포트가 올바른지 확인하거나 다른 프로그램에서 사용 중이 아닌지 확인하세요.")
