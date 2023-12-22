import os.path
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSignal, QEventLoop
from modules import MidiData
from modules.NeuralModel import GuitarInferenceDataset, LSTMModel, getLabels
import torch
import os
def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS2
    except Exception:
        base_path= os.path.abspath("")
    return os.path.join(base_path, relative_path)


class DialogBox(QDialog):

    accepted_values= pyqtSignal(tuple)
    def __init__(self, tracks, denominators, bpm):
        super().__init__()
        loadUi(resource_path('bins\\optionsDialog.ui'), self)
        self.nominator.setValue(4)
        self.denominator.addItems(denominators)
        self.denominator.setCurrentText('4')
        self.tempo.setValue(bpm)
        self.trackBox.addItems(tracks)
        self.init_ui()
    def init_ui(self):
        self.applyButton.clicked.connect(self.apply)
    def apply(self):
        instrument = self.trackBox.currentIndex()
        nominator =int(self.nominator.text())
        denominator = int(self.denominator.currentText())
        tempo = int(self.tempo.text())
        artist = self.artist.text()
        self.accepted_values.emit((instrument, nominator, denominator, tempo, artist))
        self.close()
class MyMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        loadUi(resource_path('bins\\MainWindow.ui'), self)  # Load your UI file created in Qt Designer
        self.init_ui()
        self.model = torch.load(resource_path('bins\\sample_model.pth'))
        self.model.eval()
    def init_ui(self):
        self.loadButton.clicked.connect(self.load_file)
        self.generateButton.clicked.connect(self.runOptionsDialog)


        # Initially hide the generate and save buttons
        self.generateButton.hide()



    def load_file(self):



        fname, _ = QFileDialog.getOpenFileName(self, "Open File", filter="Midi Files (*.mid)")

        if fname:
            self.generateButton.show()
            self.song=MidiData.Song(fname)

            self.instructionField.setText('<p align="center"><span style=" font-family:"MS Shell Dlg 2"; font-size:12pt;">You Can generate the Tablature </span></p>')

    def generate_output(self):


        instrument, nominator, denominator, bpm, artist = self.optionValues
        df = self.song.generateDataFrame(instrument)
        track = MidiData.Track(df,bpm, artist, (nominator, denominator))
        if not MidiData.correctCheck(track.dataframe):
            choice = self.pitch_message()

            if choice:
                track.removeWrong()
            else:
                track.pitchShiftWrong()
        features = GuitarInferenceDataset(track.getEncodedVectors(), self.model.sequence_size)
        labels = getLabels(self.model, features)
        track.assignLabels(labels)
        path, _ = QFileDialog.getSaveFileName(self, 'Save GuitarPro File', "NeuralTab", "GuitarPro 5 files (*.gp5)")
        if path:
            track.export(path)
            self.loadButton.show()
            self.song = None
            self.instructionField.setText(
                '<p align="center"><span style=" font-family:"MS Shell Dlg 2"; font-size:12pt;">File generated!</span></p>')



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
    def runOptionsDialog(self):
        options = DialogBox(self.song.instruments, list(map(str,MidiData.noteTicks.keys())), self.song.bpm)
        options.accepted_values.connect(self.getOptionsFromDialog)
        options.exec_()


    def getOptionsFromDialog(self, optionValues):
        self.optionValues = optionValues
        self.generate_output()

def run_app():
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_app()
