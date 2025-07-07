from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSizePolicy, QMessageBox, QFrame, QScrollArea
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt, QThread, Signal
import sys
import os
import json
import base64
import socket
from functools import partial


# 🔥 스타일시트
STYLESHEET = """
QWidget {
    background-color: #f58220;
    color: #000000;
    font-family: 'Malgun Gothic', '맑은 고딕', sans-serif;
}
QFrame#outer_frame {
    background-color: #f58220;
    border: 2px solid #993d00;
    border-radius: 12px;
    padding: 10px;
}
QLabQLabel#logo_label {
    background-color: #fef1dc;
    border: 6px solid #993d00;
    border-radius: 12px;
    padding: 8px;
}
QLabel#image_label {
    background-color: #fef1dc;
    border: 6px solid #993d00;
    border-radius: 12px;
    padding: 12px;
    color: #000000;
}
QLabel#info_label {
    background-color: #fff3e0;
    border: 2px solid #993d00;
    border-radius: 6px;
    padding: 12px;
    font-size: 16px;
    color: #000000;
}
QPushButton {
    background-color: #e74c3c;
    color: white;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}
QPushButton:hover {
    background-color: #c0392b;
}
QScrollArea {
    border: none;
    background-color: #ffe0b2;
}
QScrollBar:vertical {
    border: none;
    background: #f58220;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #993d00;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


# 서버로부터 데이터를 수신하는 별도의 스레드
class ReceiverThread(QThread):
    data_received = Signal(dict)
    connection_status = Signal(str)

    def __init__(self, host, port, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.running = True
        self.socket = None

    def run(self):
        self.connection_status.emit("서버 연결 시도 중...")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connection_status.emit(f"서버 연결 성공: {self.host}:{self.port}")
            buffer = ""
            while self.running:
                data = self.socket.recv(4096)
                if not data:
                    self.connection_status.emit("서버 연결 끊김.")
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        self.data_received.emit(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # 잘못된 JSON 데이터는 무시
        except socket.error as e:
            self.connection_status.emit(f"서버 연결 실패: {e}")
        finally:
            if self.socket: self.socket.close()
            self.running = False

    def stop(self):
        self.running = False
        if self.socket: self.socket.close()
        self.wait()

    def send_data(self, data):
        if self.socket:
            try:
                message = json.dumps(data) + '\n'
                self.socket.sendall(message.encode('utf-8'))
                print(f"데이터 전송 성공: {data}")
            except Exception as e:
                print(f"데이터 전송 오류: {e}")
        else:
            print("소켓이 연결되지 않아 데이터를 전송할 수 없습니다.")


class ClickableCard(QFrame):
    """클릭 가능한 카드 위젯"""
    clicked = Signal()

    def __init__(self, index, time, place):
        super().__init__()
        self.index = index
        self.setObjectName("clickable_card")  # 고유 objectName 설정
        self.setFrameShape(QFrame.StyledPanel)  # 프레임 모양 다시 설정
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)  # 내부 여백 설정
        layout.setSpacing(5)

        cam_label = QLabel(f"EVENT {index + 1}")
        cam_label.setStyleSheet("font-weight: bold; color: #fef1dc;")
        time_label = QLabel(time)
        place_label = QLabel(place)
        place_label.setStyleSheet("font-size: 11px; color: #bdc3c7;")

        layout.addWidget(cam_label)
        layout.addWidget(time_label)
        layout.addWidget(place_label)
        self.setMinimumHeight(80)  # 최소 높이 설정

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, is_selected):
        if is_selected:
            self.setStyleSheet("""
                #clickable_card {
                    background-color: #f58220;          /* ✅ 주황색 배경 */
                    border: 2px solid #993d00;          /* ✅ 불갈색 테두리 */
                    border-radius: 12px;
                    padding: 8px;
                    color: #000000;
                }
            """)
        else:
            self.setStyleSheet("""
                #clickable_card {
                    background-color: #f58220;          /* ✅ 선택 안된 것도 주황색 */
                    border: 2px solid #993d00;
                    border-radius: 12px;
                    padding: 8px;
                    color: #000000;
                }
                #clickable_card:hover {
                    background-color: #ffa64d;          /* ✅ hover 시 밝은 주황 강조 */
                }
            """)

class FireAlertUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("소방서")
        self.setGeometry(100, 100, 1920, 1080)
        self.setStyleSheet(STYLESHEET)

        self.event_data_list = []
        self.card_widgets = []
        self.selected_card_index = -1

        self.setup_ui()
        self.init_network()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        title = QLabel("\U0001F6A8 소방서")
        title.setFont(QFont("Malgun Gothic", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        outer_frame = QFrame()
        outer_frame.setObjectName("outer_frame")
        outer_layout = QHBoxLayout(outer_frame)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(10)

        # 1. 왼쪽 패널 (로고 + 카드 리스트)
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(10)
        left_panel_layout.setAlignment(Qt.AlignTop)  # 상단 정렬

        # police.jpg 이미지 추가
        police_logo_h_layout = QHBoxLayout()  # 로고를 위한 수평 레이아웃
        police_logo_h_layout.addStretch(1)  # 좌측에 스트레치 추가
        self.police_logo_label = QLabel()
        self.police_logo_label.setObjectName("logo_label")  # ✅ 스타일시트와 매칭되는 이름 설정!

        script_dir = os.path.dirname(__file__)
        police_image_path = os.path.join(script_dir, "fire_station.png")

        if os.path.exists(police_image_path):
            pixmap = QPixmap(police_image_path)
            self.police_logo_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.police_logo_label.setAlignment(Qt.AlignCenter)  # QLabel 내부에서 가운데 정렬
            self.police_logo_label.setFixedSize(300, 300)  # 로고 고정 크기
        else:
            self.police_logo_label.setText("FIRE STATION")
            self.police_logo_label.setAlignment(Qt.AlignCenter)  # QLabel 내부에서 가운데 정렬
            self.police_logo_label.setFixedSize(300, 300)  # 로고 고정 크기
            print(f"경고: {police_image_path} 파일을 찾을 수 없습니다. 텍스트로 표시합니다.")
        police_logo_h_layout.addWidget(self.police_logo_label)
        police_logo_h_layout.addStretch(1)  # 우측에 스트레치 추가
        left_panel_layout.addLayout(police_logo_h_layout)

        # 카드 리스트 스크롤 영역
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)

        left_panel_layout.addWidget(scroll_area, 1)  # 스크롤 영역이 남은 공간을 차지하도록 stretch

        outer_layout.addLayout(left_panel_layout, 2)  # 좌측 패널 전체의 stretch 조정 (2)

        # 2. 중앙 영역 (이미지 + 하단 정보)
        center_layout = QVBoxLayout()
        center_layout.setSpacing(10)

        self.image_label = QLabel("좌측 목록에서 이벤트를 선택하세요.")
        self.image_label.setObjectName("image_label")
        self.image_label.setStyleSheet("""
                        #image_label {
                            background-color: #fef1dc;
                            border: 6px solid #993d00;
                            border-radius: 12px;
                            padding: 12px;
                            color: #000000;
                        }
                    """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)  # 이미지 콘텐츠를 라벨 크기에 맞춰 스케일링
        self.image_label.setMinimumSize(600, 450)  # 최소 크기 유지
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # 레이아웃이 크기 힌트 무시
        font = self.image_label.font()
        font.setPointSize(16)
        self.image_label.setFont(font)
        center_layout.addWidget(self.image_label, 1)  # Stretch factor

        image_bottom_layout = QHBoxLayout()
        self.info_label = QLabel("화재 발생 시간 | 화재 발생 장소 | 기타정보")
        self.info_label.setObjectName("info_label")
        self.info_label.setFixedHeight(60)

        self.action_button = QPushButton("출동 완료")
        self.action_button.setFixedSize(120, 60)
        self.action_button.clicked.connect(self.show_dispatch_popup)

        image_bottom_layout.addWidget(self.info_label, 1)
        image_bottom_layout.addWidget(self.action_button)
        center_layout.addLayout(image_bottom_layout)

        outer_layout.addLayout(center_layout, 6)
        main_layout.addWidget(outer_frame, 1)

        self.statusLabel = QLabel("상태: 초기화")
        self.statusLabel.setFixedHeight(20)
        main_layout.addWidget(self.statusLabel, 0, Qt.AlignRight)

    def init_network(self):
        self.receiver_thread = ReceiverThread('10.10.20.109', 10008, self)
        self.receiver_thread.data_received.connect(self.handle_server_data)
        self.receiver_thread.connection_status.connect(self.update_status)
        self.receiver_thread.start()

    def handle_server_data(self, json_data):
        self.update_status(f"데이터 수신: {json_data.get('time')}")
        self.event_data_list.append(json_data)
        self.add_event_card(len(self.event_data_list) - 1)

        # ▼▼▼ [추가] JSON 데이터를 파일로 저장하는 부분 ▼▼▼
        try:
            script_dir = os.path.dirname(__file__)
            file_path = os.path.join(script_dir, "police.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {file_path}에 저장되었습니다.")
        except Exception as e:
            print(f"JSON 파일 저장 실패: {e}")
        # ▲▲▲ [추가] JSON 데이터를 파일로 저장하는 부분 ▲▲▲

    def add_event_card(self, index):
        event_data = self.event_data_list[index]
        card = ClickableCard(index, event_data.get('time', ''), event_data.get('place', ''))
        card.clicked.connect(partial(self.display_event_info, index))
        self.scroll_layout.insertWidget(0, card)  # 새 카드를 맨 위에 추가 (시각적 순서)
        self.card_widgets.append(card)  # card_widgets에는 event_data_list와 동일한 인덱스 순서로 추가

    def display_event_info(self, index):
        self.selected_card_index = index
        for card in self.card_widgets:
            card.set_selected(card.index == index);

        event_data = self.event_data_list[index]
        img_base64 = event_data.get("img", "")
        if img_base64:
            try:
                pixmap = QPixmap()
                pixmap.loadFromData(base64.b64decode(img_base64))
                if not pixmap.isNull():
                    self.image_label.setPixmap(pixmap.scaled(
                        self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    self.image_label.setText("이미지 데이터 손상 또는 형식 오류")
            except Exception as e:
                self.image_label.setText(f"이미지 표시 오류: {e}")
        else:
            self.image_label.setText("이미지 없음")

        self.info_label.setText(f"<b>시간:</b> {event_data.get('time', '-')} | <b>장소:</b> {event_data.get('place', '-')}")

    def show_dispatch_popup(self):
        if self.selected_card_index == -1:
            QMessageBox.warning(self, "알림", "처리할 이벤트를 먼저 선택하세요.")
            return

        msg_box = QMessageBox(self)
        msg_box.setStyleSheet("background-color: #34495e; color: black;")
        msg_box.setWindowTitle("출동 완료")
        msg_box.setText("출동을 완료하시겠습니까?\n완료 시 해당 이벤트는 목록에서 제거됩니다.")
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)

        if msg_box.exec() == QMessageBox.Yes:
            # 서버로 JSON 데이터 전송
            response_data = {
                "signal": "police_response_completed",
                "message": "출동합니다."
            }
            self.receiver_thread.send_data(response_data)
            self.remove_event()

    def remove_event(self):
        if self.selected_card_index != -1:
            # 데이터 및 위젯 제거
            card_to_remove = self.card_widgets[self.selected_card_index]
            self.scroll_layout.removeWidget(card_to_remove)  # 레이아웃에서 위젯 제거
            card_to_remove.setParent(None)  # 위젯의 부모를 None으로 설정하여 UI에서 완전히 분리
            card_to_remove.deleteLater()
            self.scroll_layout.update()  # 레이아웃 강제 업데이트

            # event_data_list와 card_widgets에서 해당 항목 제거
            self.event_data_list.pop(self.selected_card_index)
            self.card_widgets.pop(self.selected_card_index)

            # 남아있는 카드들의 인덱스 업데이트 및 시그널 재연결
            for i, card in enumerate(self.card_widgets):
                card.index = i  # card.index 업데이트
                card.clicked.disconnect()  # 기존 시그널 연결 해제
                card.clicked.connect(partial(self.display_event_info, i))  # 새로운 인덱스로 시그널 재연결

            self.reset_main_view()
            self.update_status("이벤트 처리 완료.")

    def reset_main_view(self):
        self.image_label.clear()
        self.image_label.setText("좌측 목록에서 이벤트를 선택하세요.")
        self.info_label.setText("화재 발생 시간 | 화재 발생 장소 | 기타정보")
        self.selected_card_index = -1
        for card in self.card_widgets:
            card.set_selected(False)

    def update_status(self, message):
        self.statusLabel.setText(f"상태: {message}")

    def closeEvent(self, event):
        self.receiver_thread.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FireAlertUI()
    window.show()
    sys.exit(app.exec())