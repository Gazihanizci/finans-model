import json
import sys
from dataclasses import dataclass
from typing import List

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QMessageBox,
)

from src.advisor import generate_insight
from src.config import SYMBOLS
from src.download_data import download_all
from src.predict import predict_symbol
from src.train import train_all


@dataclass
class JobResult:
    outputs: List[dict]
    message: str


class Worker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job: str, symbols: List[str] | None = None) -> None:
        super().__init__()
        self.job = job
        self.symbols = symbols or []

    def run(self) -> None:
        try:
            if self.job == "download":
                download_all()
                self.finished.emit(JobResult(outputs=[], message="Veri indirme tamamlandı."))
                return
            if self.job == "train":
                train_all()
                self.finished.emit(JobResult(outputs=[], message="Eğitim tamamlandı."))
                return
            if self.job == "predict":
                results = []
                for sym in self.symbols:
                    pred = predict_symbol(sym)
                    insight = generate_insight(pred)
                    out = {
                        "date": pred["date"],
                        "symbol": pred["symbol"],
                        "forecast": pred["forecast"],
                        "risk": pred["risk"],
                        "insight": insight,
                    }
                    results.append(out)
                self.finished.emit(JobResult(outputs=results, message="Tahmin üretildi."))
                return
            raise RuntimeError("Bilinmeyen iş.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FX Assistant (Qt5)")
        self.setMinimumSize(900, 600)

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)

        self.btn_download = QPushButton("Veri indir")
        self.btn_train = QPushButton("Model eğit")
        left_layout.addWidget(self.btn_download)
        left_layout.addWidget(self.btn_train)

        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Tahmin"))

        self.radio_all = QRadioButton("Tümü")
        self.radio_one = QRadioButton("Tek")
        self.radio_all.setChecked(True)
        left_layout.addWidget(self.radio_all)
        left_layout.addWidget(self.radio_one)

        self.symbol_box = QComboBox()
        self.symbol_box.addItems(list(SYMBOLS.keys()))
        self.symbol_box.setEnabled(False)
        left_layout.addWidget(self.symbol_box)

        self.btn_predict = QPushButton("Tahmin üret")
        left_layout.addWidget(self.btn_predict)

        left_layout.addStretch(1)

        layout.addWidget(left, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Çıktılar"))

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Segoe UI", 10))
        right_layout.addWidget(self.output, 1)

        layout.addWidget(right, 2)

        self.radio_all.toggled.connect(self._toggle_mode)
        self.btn_download.clicked.connect(self._download)
        self.btn_train.clicked.connect(self._train)
        self.btn_predict.clicked.connect(self._predict)

    def _toggle_mode(self) -> None:
        self.symbol_box.setEnabled(self.radio_one.isChecked())

    def _set_busy(self, busy: bool) -> None:
        self.btn_download.setEnabled(not busy)
        self.btn_train.setEnabled(not busy)
        self.btn_predict.setEnabled(not busy)
        self.radio_all.setEnabled(not busy)
        self.radio_one.setEnabled(not busy)
        self.symbol_box.setEnabled(not busy and self.radio_one.isChecked())

    def _start_job(self, job: str, symbols: List[str] | None = None) -> None:
        self._set_busy(True)
        self.thread = QThread()
        self.worker = Worker(job, symbols)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._job_done)
        self.worker.failed.connect(self._job_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _download(self) -> None:
        self._start_job("download")

    def _train(self) -> None:
        self._start_job("train")

    def _predict(self) -> None:
        if self.radio_all.isChecked():
            symbols = list(SYMBOLS.keys())
        else:
            symbols = [self.symbol_box.currentText()]
        self._start_job("predict", symbols)

    def _job_done(self, result: JobResult) -> None:
        self._set_busy(False)
        if result.message:
            self.output.append(result.message)
        for item in result.outputs:
            self.output.append(json.dumps(item, ensure_ascii=False))
        if result.outputs:
            self.output.append("")

    def _job_failed(self, msg: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Hata", msg)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
