"""
Academic demonstration application for the AI-Based Skin Cancer Detection project.

This public version:
- Does not connect to Firebase.
- Does not collect, store, or upload personal information.
- Does not include credentials, datasets, or the TensorFlow Lite model.
"""

import sys
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QBrush, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
)


MODEL_PATH = "skin_model.tflite"
BACKGROUND_PATH = "background.png"


class SkinAIDemo(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedSize(640, 480)
        self.setWindowTitle("SKIN AI — Academic Demonstration")

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.last_frame = None

        self.set_background(BACKGROUND_PATH)
        self.load_model()

        self.stacked_widget = QStackedWidget()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked_widget)

        self.create_welcome_screen()
        self.create_camera_screen()
        self.create_result_screen()

        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.camera_screen)
        self.stacked_widget.addWidget(self.result_screen)

    def set_background(self, image_path):
        if not os.path.exists(image_path):
            return

        palette = QPalette()
        image = QPixmap(image_path)
        scaled_image = image.scaled(
            self.size(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
        palette.setBrush(QPalette.Window, QBrush(scaled_image))
        self.setPalette(palette)

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            return

        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as error:
            print(f"Model loading error: {error}")

    def create_welcome_screen(self):
        self.welcome_screen = QWidget()
        layout = QVBoxLayout(self.welcome_screen)

        layout.addStretch(1)

        title = QLabel("SKIN AI")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white;")

        subtitle = QLabel(
            "Academic preliminary screening prototype\n"
            "No personal information is collected or stored."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            "color: white; background: rgba(0, 0, 0, 120); padding: 12px;"
        )

        disclaimer = QLabel(
            "This prototype does not replace dermatologist diagnosis, "
            "clinical examination, or biopsy."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setAlignment(Qt.AlignCenter)
        disclaimer.setStyleSheet(
            "color: #ffffff; background: rgba(140, 35, 45, 180); padding: 10px;"
        )

        start_button = QPushButton("START DEMONSTRATION")
        start_button.setFixedWidth(300)
        start_button.clicked.connect(self.start_camera_session)
        start_button.setStyleSheet(
            """
            QPushButton {
                background-color: #198fe3;
                color: white;
                border-radius: 10px;
                padding: 14px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1676b9;
            }
            """
        )

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(disclaimer)
        layout.addWidget(start_button, alignment=Qt.AlignCenter)
        layout.addStretch(1)

    def create_camera_screen(self):
        self.camera_screen = QWidget()
        layout = QVBoxLayout(self.camera_screen)

        heading = QLabel("CAMERA PREVIEW")
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("color: white; font-weight: bold;")

        self.camera_view = QLabel()
        self.camera_view.setFixedSize(480, 320)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(
            "background: black; border: 3px solid rgba(255,255,255,150);"
        )

        hint = QLabel("Press SPACE to process the current frame.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(
            "color: white; background: rgba(0,0,0,120); padding: 8px;"
        )

        privacy_note = QLabel("Frames are processed locally and are not saved or uploaded.")
        privacy_note.setAlignment(Qt.AlignCenter)
        privacy_note.setStyleSheet("color: white;")

        layout.addStretch(1)
        layout.addWidget(heading)
        layout.addWidget(self.camera_view, alignment=Qt.AlignCenter)
        layout.addWidget(hint)
        layout.addWidget(privacy_note)
        layout.addStretch(1)

        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, 640)
        self.camera.set(4, 480)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_stream)

    def create_result_screen(self):
        self.result_screen = QWidget()
        layout = QVBoxLayout(self.result_screen)

        self.result_title = QLabel("Model Output")
        self.result_title.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_title.setAlignment(Qt.AlignCenter)

        images_layout = QHBoxLayout()
        self.original_image_label = QLabel()
        self.visual_heatmap_label = QLabel()

        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.visual_heatmap_label.setAlignment(Qt.AlignCenter)

        images_layout.addWidget(self.original_image_label)
        images_layout.addWidget(self.visual_heatmap_label)

        explanation = QLabel(
            "Left: processed image   |   Right: visual heatmap\n"
            "The visual heatmap is an interface visualization and not a clinical explanation."
        )
        explanation.setWordWrap(True)
        explanation.setAlignment(Qt.AlignCenter)
        explanation.setStyleSheet("color: white; background: rgba(0,0,0,120); padding: 8px;")

        restart_button = QPushButton("NEW DEMONSTRATION")
        restart_button.clicked.connect(self.return_to_start)
        restart_button.setStyleSheet(
            """
            QPushButton {
                background-color: #20a85a;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #187b42;
            }
            """
        )

        layout.addStretch(1)
        layout.addWidget(self.result_title)
        layout.addLayout(images_layout)
        layout.addWidget(explanation)
        layout.addWidget(restart_button, alignment=Qt.AlignCenter)
        layout.addStretch(1)

    def start_camera_session(self):
        if self.interpreter is None:
            self.result_title.setText(
                "Model file not found.\n"
                "Keep skin_model.tflite locally; do not upload it to the public repository."
            )
            self.stacked_widget.setCurrentIndex(2)
            return

        self.stacked_widget.setCurrentIndex(1)

        if not self.timer.isActive():
            self.timer.start(30)

    def update_camera_stream(self):
        success, frame = self.camera.read()

        if not success:
            return

        self.last_frame = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = QImage(
            rgb_frame.data,
            rgb_frame.shape[1],
            rgb_frame.shape[0],
            rgb_frame.shape[1] * 3,
            QImage.Format_RGB888,
        )

        self.camera_view.setPixmap(
            QPixmap.fromImage(image).scaled(
                480,
                320,
                Qt.KeepAspectRatio,
            )
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and self.stacked_widget.currentIndex() == 1:
            self.process_current_frame()

    def process_current_frame(self):
        if self.last_frame is None or self.interpreter is None:
            return

        original_frame = self.last_frame.copy()

        input_height = self.input_details[0]["shape"][1]
        input_width = self.input_details[0]["shape"][2]

        processed_image = cv2.resize(original_frame, (input_width, input_height))
        input_data = np.expand_dims(processed_image.astype(np.float32), axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"]).flatten()

        # The original project model uses Benign as index 0 and Malignant as index 1.
        if len(output) >= 2:
            benign_probability = float(output[0])
            malignant_probability = float(output[1])

            if benign_probability < 0.50:
                prediction = "Malignant"
                confidence = malignant_probability
            else:
                prediction = "Benign"
                confidence = benign_probability
        else:
            malignant_probability = float(output[0])
            prediction = "Malignant" if malignant_probability >= 0.50 else "Benign"
            confidence = malignant_probability if prediction == "Malignant" else 1 - malignant_probability

        grayscale = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
        visual_heatmap = cv2.addWeighted(processed_image, 0.5, color_map, 0.5, 0)

        self.show_results(processed_image, visual_heatmap, prediction, confidence)

    def show_results(self, original, heatmap, prediction, confidence):
        def to_pixmap(image):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(
                rgb_image.data,
                rgb_image.shape[1],
                rgb_image.shape[0],
                rgb_image.shape[1] * 3,
                QImage.Format_RGB888,
            )
            return QPixmap.fromImage(q_image)

        self.original_image_label.setPixmap(
            to_pixmap(original).scaled(200, 200, Qt.KeepAspectRatio)
        )
        self.visual_heatmap_label.setPixmap(
            to_pixmap(heatmap).scaled(200, 200, Qt.KeepAspectRatio)
        )

        colour = "#e74c3c" if prediction == "Malignant" else "#2ecc71"
        self.result_title.setText(
            f"Academic model output: {prediction} ({confidence:.1%})"
        )
        self.result_title.setStyleSheet(
            f"color: {colour}; background: rgba(0,0,0,150); "
            "padding: 10px; border-radius: 5px;"
        )

        self.timer.stop()
        self.stacked_widget.setCurrentIndex(2)

    def return_to_start(self):
        self.stacked_widget.setCurrentIndex(0)


if __name__ == "__main__":
    application = QApplication(sys.argv)
    window = SkinAIDemo()
    window.show()
    sys.exit(application.exec_())
