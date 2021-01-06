import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from interfaz import Ui_Dialog
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import translators as ts

from partes_terminadas import hasForbidden

def load_model(model_path):
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

modelo = load_model("comments.h5")

def predict(comment):
    return modelo.predict(np.array([comment]))[0][0]
app = QtWidgets.QApplication(sys.argv)

Dialog = QtWidgets.QDialog()
ui = Ui_Dialog()
ui.setupUi(Dialog)
Dialog.show()

def get_comment():
    comment_ru = ui.textEdit.toPlainText()
    comment = ts.translate_html(comment_ru, translator=ts.google, to_language='en', translator_params={})
    text = ""
    prob = predict(comment) * 100
    if(hasForbidden(comment_ru)):
        text += "Запрещенное слово!!!\n"
    text += "Вероятность хорошего коммента: {:.2f}%\n".format(prob)
    if prob > 60:
        text += "скорее всего положительный"
    else:
        text += "скорее всего отрицательный"
    ui.label.setText(text)

ui.pushButton.clicked.connect(get_comment)
sys.exit(app.exec_())
