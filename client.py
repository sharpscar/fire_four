from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QDialog, QPushButton
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, Qt, QThread, Signal
import sys, cv2, torch, torch.nn as nn, socket, json, base64, os
from torchvision import transforms
from PIL import Image
from ui_mainwindow import Ui_MainWindow
import time


# base64 to QPixmap
def base64_to_qpixmap(base64_string):
    try:
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]

        base64_string = base64_string.replace("\n", "").replace("\r", "")
        image_data = base64.b64decode(base64_string)
        image = QImage.fromData(image_data)

        if image.isNull():
            print("QImage 변환 실패: 유효하지 않은 이미지 데이터")
            return None

        return QPixmap.fromImage(image)
    except Exception as e:
        print(f"base64_to_qpixmap 변환 중 오류: {e}")
        return None


# 이미지 파일로 저장하기
def save_image_from_base64(base64_string, filename):
    try:
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]

        image_data = base64.b64decode(base64_string)
        save_directory = "/home/yang/PycharmProjects/PythonProject/img_03"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        file_path = os.path.join(save_directory, filename)

        with open(file_path, 'wb') as image_file:
            image_file.write(image_data)
        print(f"이미지가 '{file_path}'로 저장되었습니다.")
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {e}")


# 클릭 시 대화상자에 이미지 띄우기
def show_image_in_dialog(image_pixmap):
    dialog = QDialog()
    dialog.setWindowTitle("화재 이미지")

    # 레이아웃 생성 및 중앙 정렬
    layout = QVBoxLayout(dialog)
    layout.setAlignment(Qt.AlignCenter)  # 중앙 정렬 추가

    # 이미지 라벨 생성
    image_label = QLabel()
    image_label.setPixmap(image_pixmap.scaled(600, 600, Qt.KeepAspectRatio))  # 크기 조정
    layout.addWidget(image_label)

    # 확인 버튼
    close_button = QPushButton("확인")
    close_button.clicked.connect(dialog.accept)  # 확인 버튼 클릭 시 대화상자 닫기
    layout.addWidget(close_button)

    dialog.exec()  # 대화상자 실행


# 가로형 결과 위젯
def create_fire_result_widget(camera, address, time_str, pixmap):
    widget = QWidget()
    widget.setMinimumHeight(80)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(5, 5, 5, 5)
    layout.setSpacing(20)

    # 각 항목 생성
    time_label = QLabel(time_str)
    cam_label = QLabel(camera)
    addr_label = QLabel(address)
    img_label = QLabel()

    # 이미지 크기 지정
    img_label.setFixedSize(120, 90)
    img_label.setPixmap(pixmap.scaled(120, 90, Qt.KeepAspectRatio))

    # 스타일 및 너비 조정
    time_label.setStyleSheet("color: white; font-size: 14px;")
    cam_label.setStyleSheet("color: white; font-size: 14px;")
    addr_label.setStyleSheet("color: white; font-size: 14px;")

    # 이미지 클릭 이벤트 추가
    img_label.mousePressEvent = lambda event: show_image_in_dialog(pixmap)  # 클릭 시 이미지를 대화상자에 띄우기

    time_label.setFixedWidth(150)
    cam_label.setFixedWidth(80)
    addr_label.setFixedWidth(600)

    time_label.setAlignment(Qt.AlignCenter)
    cam_label.setAlignment(Qt.AlignCenter)
    addr_label.setAlignment(Qt.AlignCenter)

    # 위젯 추가
    layout.addWidget(time_label)
    layout.addWidget(cam_label)
    layout.addWidget(addr_label)
    layout.addWidget(img_label)

    return widget


# 수신 스레드
class ReceiverThread(QThread):
    fire_result_received = Signal(list)

    def __init__(self, socket, parent=None):
        super().__init__(parent)
        self.socket = socket
        self.running = True

    def run(self):
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8", errors="ignore")  # 또는 errors="replace"
                while '\n' in buffer:
                    msg, buffer = buffer.split('\n', 1)
                    try:
                        json_data = json.loads(msg)
                        print("수신된 JSON:", json_data)
                        if json_data.get("signal") == "cam_result":
                            cam_info = json_data.get("cam_info", {})

                            if isinstance(cam_info, dict):
                                # 딕셔너리 단일 객체인 경우
                                self.fire_result_received.emit([cam_info])
                            elif isinstance(cam_info, list):
                                # 리스트로 여러 개가 온 경우
                                self.fire_result_received.emit(cam_info)
                            else:
                                print(" cam_info 타입 오류:", type(cam_info))
                                self.fire_result_received.emit([])  # fallback
                    except json.JSONDecodeError as e:
                        print(f"JSON 디코딩 오류: {e}")
            except socket.timeout:
                continue
            except Exception as e:
                print("수신 중 오류 발생:", e)
                break

    def stop(self):
        self.running = False
        self.wait()


# 모델 정의
class FireNetImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# 메인 윈도우
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("화재 감지 AI")
        self.ui.stackedWidget.setCurrentIndex(0)

        self.cap1 = cv2.VideoCapture(2)
        self.cap3 = cv2.VideoCapture(4)
        self.current_frame_cam1 = None
        self.current_frame_cam3 = None
        self.current_camera_index = None
        self.cam1_fire = False
        self.cam3_fire = False
        self.fire_sent_cam1 = False
        self.fire_sent_cam3 = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FireNetImproved().to(self.device)
        self.model.load_state_dict(torch.load("fire_model_improved_finetuned_best.pth", map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

        self.HOST = '10.10.20.109'
        self.PORT = 10008
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_server()

        self.receiver_thread = ReceiverThread(self.socket)
        self.receiver_thread.fire_result_received.connect(self.add_fire_widget)
        self.receiver_thread.start()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.ui.scrollAreaWidgetContents_3.setLayout(layout)

        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self.update_frame_cam1)
        self.timer1.start(30)
        self.timer3 = QTimer(self)
        self.timer3.timeout.connect(self.update_frame_cam3)
        self.timer3.start(30)

        self.auto_timer_cam1 = QTimer(self)
        self.auto_timer_cam1.timeout.connect(self.analyze_frame_cam1)
        self.auto_timer_cam1.start(100)
        self.auto_timer_cam3 = QTimer(self)
        self.auto_timer_cam3.timeout.connect(self.analyze_frame_cam3)
        self.auto_timer_cam3.start(100)

        self.ui.check_fire_status.clicked.connect(self.goto_fire_result_page)
        self.ui.camara1_btn.clicked.connect(self.set_camera1_view)
        self.ui.camara2_btn.clicked.connect(self.set_camera3_view)
        self.ui.back_btn.clicked.connect(self.go_back_to_main)
        self.ui.check_fire_cam1_btn_2.clicked.connect(lambda: self.send_manual_signal("cam01"))
        self.ui.check_fire_cam2_btn_2.clicked.connect(lambda: self.send_manual_signal("cam03"))
        self.ui.back_btn_2.clicked.connect(self.go_back_to_main_2)

    def connect_to_server(self):
        try:
            self.socket.connect((self.HOST, self.PORT))
            self.socket.settimeout(1.0)
            print(" 서버 연결 성공")
        except Exception as e:
            print(" 서버 연결 실패:", e)

    def goto_fire_result_page(self):
        self.clear_scroll_area()
        self.ui.stackedWidget.setCurrentIndex(1)

    def clear_scroll_area(self):
        layout = self.ui.scrollArea.widget().layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

    def add_fire_widget(self, cam_info_list: list):
        try:
            print(f"서버로부터 받은 cam_info_list: {cam_info_list}")  # 디버그: 수신된 카메라 정보 확인
            # cam_info_list가 빈 리스트가 아니고, 실제로 데이터가 들어있다면
            if isinstance(cam_info_list, list) and len(cam_info_list) > 0:
                for cam_info in cam_info_list:
                    # cam_info가 딕셔너리로 되어 있을 때 처리
                    b64_img = cam_info.get("picture", "")  # 이미지 데이터
                    camera = cam_info.get("camera", "Unknown")  # 카메라 이름
                    address = cam_info.get("address", "Unknown")  # 주소
                    time_str = cam_info.get("time", "Unknown")  # 시간 (옵션)

                    filename = f"{camera}_{time_str.replace(':', '-')}.jpg"
                    save_image_from_base64(b64_img, filename)

                    print(f"카메라 이름: {camera}, 주소: {address}, 시간: {time_str}")  # 디버그: 받은 정보 확인

                    # base64 이미지를 QPixmap으로 변환
                    pixmap = base64_to_qpixmap(b64_img)
                    if not pixmap:
                        print(f"{camera} → base64 → QPixmap 변환 실패")
                        continue

                    # 결과 위젯을 UI에 추가
                    widget = create_fire_result_widget(camera, address, time_str, pixmap)
                    self.ui.scrollArea.widget().layout().addWidget(widget)
                    print(f" {camera} 화재 감지 결과 추가됨")  # 디버그: 화재 감지 결과 추가 여부 확인

                self.ui.scrollArea.update()
            else:
                print(" cam_info_list가 빈 리스트입니다.")
        except Exception as e:
            print(f"위젯 추가 중 오류 발생:", e)

    def send_manual_signal(self, signal_str):
        self.goto_fire_result_page()
        try:
            self.socket.sendall((json.dumps({"signal": signal_str}) + '\n').encode('utf-8'))
            print(f" 시그널 전송: {signal_str}")
        except Exception as e:
            print(" 시그널 전송 실패:", e)

    def update_frame_cam1(self):
        ret, frame = self.cap1.read()
        if ret:
            self.current_frame_cam1 = frame
            self._set_frame_to_label(self.ui.videoLabel, frame, 1)

    def update_frame_cam3(self):
        ret, frame = self.cap3.read()
        if ret:
            self.current_frame_cam3 = frame
            self._set_frame_to_label(self.ui.videoLabel2, frame, 3)

    def _set_frame_to_label(self, label, frame, cam_index):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        label.setPixmap(pixmap)  # 기본 label
        if self.ui.stackedWidget_2.currentIndex() == 1 and self.current_camera_index == cam_index:
            #비율 무시하고 꽉 채우기
            self.ui.select_camera.setPixmap(pixmap.scaled(
                self.ui.select_camera.width(),
                self.ui.select_camera.height(),
                Qt.IgnoreAspectRatio))

    def set_camera1_view(self):
        self.current_camera_index = 1
        self.ui.stackedWidget_2.setCurrentIndex(1)

    def set_camera3_view(self):
        self.current_camera_index = 3
        self.ui.stackedWidget_2.setCurrentIndex(1)

    def go_back_to_main(self):
        self.ui.stackedWidget_2.setCurrentIndex(0)
        self.current_camera_index = None

    def go_back_to_main_2(self):
        self.clear_scroll_area()
        self.ui.stackedWidget.setCurrentIndex(0)

    def analyze_frame(self, frame, camera_name):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = torch.argmax(self.model(input_tensor), dim=1).item()

        if camera_name == "cam01":
            if pred == 0 and not self.cam1_fire:
                self.cam1_fire = True
                self.send_to_server(camera_name, frame)
                # 이미지 저장
                now = time.strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{camera_name}_{now}.jpg"
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                save_image_from_base64(img_base64, filename)
            elif pred != 0 and self.cam1_fire:
                self.cam1_fire = False
        elif camera_name == "cam03":
            if pred == 0 and not self.cam3_fire:
                self.cam3_fire = True
                self.send_to_server(camera_name, frame)
                # 이미지 저
                now = time.strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{camera_name}_{now}.jpg"
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                save_image_from_base64(img_base64, filename)
            elif pred != 0 and self.cam3_fire:
                self.cam3_fire = False

        self.update_status_label()

    def analyze_frame_cam1(self):
        if self.current_frame_cam1 is not None:
            self.analyze_frame(self.current_frame_cam1, "cam01")

    def analyze_frame_cam3(self):
        if self.current_frame_cam3 is not None:
            self.analyze_frame(self.current_frame_cam3, "cam03")

    def update_status_label(self):
        status = []
        if self.cam1_fire: status.append("cam01")
        if self.cam3_fire: status.append("cam03")
        if status:
            self.ui.sensor.setStyleSheet("background-color: red; border-radius: 35px;")
            self.ui.statusLabel.setText(" 화재 감지 (" + ", ".join(status) + ")")
        else:
            self.ui.sensor.setStyleSheet("background-color: rgb(143, 240, 164); border-radius: 35px;")
            self.ui.statusLabel.setText(" 정상 상태")

    def send_to_server(self, camera_name, frame):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            data = {"signal": "fire", "camera": camera_name, "picture": img_base64}
            self.socket.sendall((json.dumps(data) + '\n').encode('utf-8'))
            print(f"🔥 서버로 화재 데이터 전송됨: {camera_name}")
        except Exception as e:
            print(" 서버 전송 실패:", e)

    def closeEvent(self, event):
        if self.cap1: self.cap1.release()
        if self.cap3: self.cap3.release()
        try:
            self.receiver_thread.stop()
            self.socket.close()
        except Exception:
            pass
        event.accept()


# 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
