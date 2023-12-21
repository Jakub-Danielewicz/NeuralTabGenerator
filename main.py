import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from modules import MidiData
from modules.NeuralModel import GuitarInferenceDataset, LSTMModel, getLabels
import torch

class MyMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        loadUi('bins/MainWindow.ui', self)  # Load your UI file created in Qt Designer
        self.init_ui()
        self.model = torch.load('bins/sample_model.pth')
        self.model.eval()
    def init_ui(self):
        self.loadButton.clicked.connect(self.load_file)
        self.generateButton.clicked.connect(self.generate_output)


        # Initially hide the generate and save buttons
        self.generateButton.hide()
        for widget in self.verticalLayout.children():
            widget.hide()
        self.generating.hide()
        self.progressBar.hide()


    def load_file(self):



        fname, _ = QFileDialog.getOpenFileName(self, "Open File", filter="Midi Files (*.mid)")

        if fname:
            self.loadButton.hide()
            self.generateButton.show()
            self.song=MidiData.Song(fname)

            print(self.song.instruments)

    def generate_output(self):
        df=self.song.generateDataFrame(0)
        track = MidiData.Track(df, self.song.bpm)
        if not MidiData.correctCheck(track.dataframe):
            choice = self.pitch_message()

            if choice:
                track.removeWrong()
            else:
                track.pitchShiftWrong()
        features = GuitarInferenceDataset(track.getEncodedVectors(), self.model.sequence_size)
        labels = getLabels(self.model, features)
        track.assignLabels(labels)
        path, _ = QFileDialog.getSaveFileName(self, 'Save GuitarPro File', "", "GuitarPro 5 files (*.gp5)")
        if path:
            track.export(path)
            self.loadButton.show()
            self.generateButton.hide()
            self.song = None




    def pitch_message(self):
        msg_box = QMessageBox()
        msg_box.setText("Notes outside of guitar tuning range detected, how should they be treated?")
        yes_button = msg_box.addButton("Remove", QMessageBox.YesRole)
        no_button = msg_box.addButton("Pitch Shift", QMessageBox.NoRole)

        msg_box.exec_()

        if msg_box.clickedButton() == yes_button:
            return True
        elif msg_box.clickedButton() == no_button:
            return False
def run_app():
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_app()
