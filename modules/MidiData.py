import pretty_midi
import pandas as pd
import math
import numpy as np
from modules import dadagp

noteTicks = {
    128: 30,
    64: 60,
    32: 120,
    16: 240,
    8: 480,
    4: 960,
    2: 1920,
    1: 3840
}

tuning = {  #słownik z wysokościami MIDI danych strun
    1: 64,
    2: 59,
    3: 55,
    4: 50,
    5: 45,
    6: 40
}
def pitchshift(value):
    while not (tuning[6] <= value <= tuning[1]+24):
        if value < tuning[6]:
            value += 12
        elif value > tuning[1]+24:
            value -= 12
    return value


def bpmToTicks(bpm: int):

    return bpm*960/60
def measureVector(end, timesig):
    step = timesig[0]*noteTicks[timesig[1]]
    return np.arange(0,end,step)

def correctCheck(data):
    max=data['pitch'].max()
    min=data['pitch'].min()
    print(min, max)
    if max > tuning[1]+24 or min < tuning[6]:
        return False
    return True

class Song:
    def __init__(self, fname):
        try:
            self.midi_data = pretty_midi.PrettyMIDI(fname)
        except Exception as e:
            print(e)
        self.bpm=math.floor(self.midi_data.estimate_tempo())
        self.instruments = [instrument.name for instrument in self.midi_data.instruments]

    def generateDataFrame(self,instrument):

        ticksPerSecond = bpmToTicks(self.bpm)

        notes = self.midi_data.instruments[instrument].notes
        data = pd.DataFrame(columns=['pitch', 'start', 'end', 'string', 'fret'])
        all_timestamps = data['start'].unique()
        for i, note in enumerate(notes):
            data.loc[i] = [int(note.pitch), round(note.start * ticksPerSecond), round(note.end * ticksPerSecond), 0, 0]
        data['pitch'] = data['pitch'].astype(int)
        return data

class Track:
    def __init__(self, dataframe, bpm, timesig=(4,4),resolution=32):
        self.dataframe = dataframe
        self.bpm = bpm
        self.timesig= timesig
        self.measures = measureVector(self.dataframe['end'].max(),timesig=self.timesig)

    def getEncodedVectors(self):
        steps = self.dataframe['start'].unique()
        sequences = []
        for timestamp in steps:
            group = self.dataframe[self.dataframe['start'] == timestamp]
            vector = np.zeros(128, dtype=np.int8)  # jest 128 nut midi
            for _, note in group.iterrows():
                if note['fret'] not in range(0, 25):
                    continue
                vector[int(note['pitch'])] = 1
            sequences.append(vector)
        return np.array(sequences)

    def assignLabels(self, labels):
        all_timestamps = self.dataframe['start'].unique()
        counter_activations = 0
        counter_no_labels = 0
        for i, timestamp in enumerate(all_timestamps):  # jest 128 nut midi
            group = self.dataframe[self.dataframe['start'] == timestamp]
            for idx, note in group.iterrows():
                pitch = note['pitch']
                for string, fret in enumerate(labels[i]):
                    if tuning[string + 1] + fret == pitch and fret != 25:
                        if self.dataframe.loc[idx, 'string'] != 0:
                            print(
                                f'2 activations at once!!, string: {string + 1}, pitch: {pitch}, {labels[i]}, already: {self.dataframe.loc[idx, "string"]}')
                            counter_activations += 1

                        self.dataframe.loc[idx, 'string'] = string + 1
                        self.dataframe.loc[idx, 'fret'] = fret.item()
                        # break
                    if string == 5 and self.dataframe.loc[idx, 'string'] == 0:
                        print(
                            f'nie znaleziono odpowiedniego labela: \n note number: {idx} pitch: {pitch}, tensor: {labels[i]}')
                        counter_no_labels += 1
                        self.dataframe.loc[idx, 'string'] = 1
                        self.dataframe.loc[idx, 'fret'] = 22


    def export(self, path):
        artist = "Jakub Danielewicz"
        downtune = "downtune:" + str(0)
        tempo = "tempo:" + str(self.bpm)
        tokens = [artist, downtune, tempo, 'start', "new_measure"]
        tokenBase = pd.concat([self.dataframe['start'], self.dataframe['end']]).unique()
        tokenBase=np.append(tokenBase, self.measures)
        tokenBase=np.unique(tokenBase)


        toEncode = np.empty((len(tokenBase), 1), dtype=list)
        toEncode = [[] for _ in range(len(tokenBase))]
        if self.dataframe.loc[0]['start'] != 0:
            tokens.append("wait:" + str(int(self.dataframe.loc[0]['start'])))
            # tokens.append("new_measure")
        for _, note in self.dataframe.iterrows():
            start = note['start']
            end = note['end']
            indices = [index for index, value in enumerate(tokenBase) if value >= start and value < end]
            for i, idx in enumerate(indices):
                if i == 0:
                    toEncode[idx].append([note['string'], note['fret'], 1])
                else:
                    toEncode[idx].append([note['string'], note['fret'], 0])


        ticksPerMeasure = self.timesig[0] * (
                    3840 / self.timesig[1])

        for idx, times in enumerate(zip(tokenBase, tokenBase[1:])):
            start, end = times
            # print(start,ticksPerMeasure)

            if start in self.measures:
                tokens.append("new_measure")
            for note in toEncode[idx]:
                tokens.append("clean0:note:s" + str(int(note[0])) + ":f" + str(int(note[1])))
                if note[2] == 0:
                    tokens.append("nfx:tie")
            tokens.append("wait:" + str(int(end - start)))

        tokens.append("end")
        dadagp.dadagp_decode(tokens, path)
        print(self.measures)
    def removeWrong(self):
        self.dataframe = self.dataframe[self.dataframe['pitch']<=tuning[1]+24]
        self.dataframe = self.dataframe[self.dataframe['pitch'] >= tuning[6]]
    def pitchShiftWrong(self):
        filtered_indices = (self.dataframe['pitch'] > tuning[1]+24) | (self.dataframe['pitch'] < tuning[6] )
        self.dataframe.loc[filtered_indices, 'pitch'] = self.dataframe.loc[filtered_indices, 'pitch'].apply(pitchshift).astype("int32")
        self.dataframe['pitch'].astype('int32')