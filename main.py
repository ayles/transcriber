import wx
import wave
import json
import numpy as np
from vosk import Model, SpkModel, KaldiRecognizer, SetLogLevel
from recasepunc import CasePuncPredictor, WordpieceTokenizer


def cosine_dist(x, y):
    nx = np.array(x)
    ny = np.array(y)
    return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)

class Form(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Portable Recognizer')

        panel = wx.Panel(self)

        vbox = wx.BoxSizer(wx.VERTICAL)

        model_text = wx.StaticText(panel, label="Choose model path:")
        self.model_picker = wx.DirPickerCtrl(panel, message="Choose model path", path="/Users/ayles/Downloads/vosk-model-small-ru-0.22")
        vbox.Add(model_text, flag=(wx.ALL & ~wx.BOTTOM) | wx.EXPAND, border=10)
        vbox.Add(self.model_picker, flag=wx.ALL | wx.EXPAND, border=10)

        speaker_model_text = wx.StaticText(panel, label="Choose speaker model path:")
        self.speaker_model_picker = wx.DirPickerCtrl(panel, message="Choose speaker model path", path="/Users/ayles/Downloads/vosk-model-spk-0.4")
        vbox.Add(speaker_model_text, flag=(wx.ALL & ~wx.BOTTOM) | wx.EXPAND, border=10)
        vbox.Add(self.speaker_model_picker, flag=wx.ALL | wx.EXPAND, border=10)

        punctuation_model_text = wx.StaticText(panel, label="Choose punctuation model path:")
        self.punctuation_file_picker = wx.FilePickerCtrl(panel, message="Choose punctuation model file", path="/Users/ayles/Downloads/vosk-recasepunc-ru-0.22/checkpoint")
        vbox.Add(punctuation_model_text, flag=(wx.ALL & ~wx.BOTTOM) | wx.EXPAND, border=10)
        vbox.Add(self.punctuation_file_picker, flag=wx.ALL | wx.EXPAND, border=10)

        audio_text = wx.StaticText(panel, label="Choose audio file:")
        self.audio_picker = wx.FilePickerCtrl(panel, message="Choose audio file", path="/Users/ayles/Downloads/index(2).wav")
        vbox.Add(audio_text, flag=(wx.ALL & ~wx.BOTTOM) | wx.EXPAND, border=10)
        vbox.Add(self.audio_picker, flag=wx.ALL | wx.EXPAND, border=10)

        self.recognized_text = wx.TextCtrl(panel, style=wx.HSCROLL | wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH)
        self.recognized_text.ClearBackground()
        vbox.Add(self.recognized_text, 1, flag=(wx.ALL & ~wx.BOTTOM) | wx.EXPAND, border=10)

        recognize_button = wx.Button(panel, label="Recognize")
        recognize_button.Bind(wx.EVT_BUTTON, self.recognize)
        vbox.Add(recognize_button, flag=wx.ALL | wx.EXPAND, border=10)

        panel.SetSizer(vbox)

        self.SetSize(width=480, height=600)

        self.Centre()
        self.Show()

    def recognize(self, ev):
        wf = wave.open(self.audio_picker.GetPath(), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            exit(1)

        model = Model(model_path=self.model_picker.GetPath())
        speaker_model = SpkModel(model_path=self.speaker_model_picker.GetPath())
        punctuation_predictor = CasePuncPredictor(checkpoint_path=self.punctuation_file_picker.GetPath())

        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetSpkModel(speaker_model)
        rec.SetWords(True)
        rec.SetPartialWords(True)

        speakers = []
        prev_speaker = None
        speakers_texts = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                speaker_index = None
                for i in range(len(speakers)):
                    if cosine_dist(speakers[i], res['spk']) < 0.5:
                        speaker_index = i
                        break
                if speaker_index is None:
                    speaker_index = len(speakers)
                    speakers.append(res['spk'])

                if speaker_index != prev_speaker:
                    speakers_texts.append([speaker_index])
                    prev_speaker = speaker_index

                results_after_punctuation = ""
                for token, case_label, punc_label in punctuation_predictor.predict(list(enumerate(punctuation_predictor.tokenize(res['text']))), lambda x: x[1]):
                    prediction = punctuation_predictor.map_punc_label(
                        punctuation_predictor.map_case_label(token[1], case_label), punc_label)
                    if token[1][0] != '#':
                        results_after_punctuation = results_after_punctuation + ' ' + prediction
                    else:
                        results_after_punctuation = results_after_punctuation + prediction

                speakers_texts[-1].append(results_after_punctuation.strip())

        self.recognized_text.Clear()
        for st in speakers_texts:
            self.recognized_text.SetDefaultStyle(wx.TextAttr(colText='#aaffaa'))
            self.recognized_text.AppendText('Speaker' + str(st[0]) + ':\n')
            self.recognized_text.SetDefaultStyle(wx.TextAttr(colText='#aaaaaa'))
            self.recognized_text.AppendText('\n'.join(st[1:]) + '\n\n')


if __name__ == '__main__':
    my_app = wx.App()
    form = Form()
    my_app.MainLoop()
