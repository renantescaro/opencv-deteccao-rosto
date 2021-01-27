import cv2
import os
import numpy as np

class Treinamento:
    def __init__(self):
        self.eigenface  = cv2.face.EigenFaceRecognizer_create()
        self.fisherface = cv2.face.FisherFaceRecognizer_create()
        self.lbph       = cv2.face.LBPHFaceRecognizer_create()


    def _get_imagem_com_id(self):
        faces = []
        ids   = []
        caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
        for caminho_imagem in caminhos:

            # pega imagem da pasta fotos e converte para cinza
            imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)
            # pega o id no nome da imagem
            id = int(os.path.split(caminho_imagem)[-1].split('.')[1])

            ids.append(id)
            faces.append(imagem_face)
        return np.array(ids), faces

    def iniciar(self):
        ids, faces = self._get_imagem_com_id()

        # treina com as imagens e grava arquivo de aprendizado
        self.eigenface.train(faces, ids)
        self.eigenface.write('classificador_eigen.yml')
        self.fisherface.train(faces, ids)
        self.fisherface.write('classificador_disherface.yml')
        self.lbph.train(faces, ids)
        self.lbph.write('classificador_lbph.yml')