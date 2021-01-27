import cv2
import numpy as np

class CapturaFaces:
    def __init__(self):
        # parametro 0 para webcam integrada / usb
        self.camera = cv2.VideoCapture('http://192.168.0.200:8080/?action=stream')
        self.classificador_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.classificador_olhos = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.amostra = 1
        self.numero_amostras = 25
        self.largura_foto = 220
        self.altura_foto  = 220


    def iniciar(self, id_pessoa):
        while (True):
            conectado, imagem = self.camera.read()
            imagem_cinza      = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            faces_detectadas  = self.classificador_face.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150, 150))

            # retangulos em faces detectadas
            for (x, y, l, a) in faces_detectadas:
                cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)

                # verificar se tem olhos dentro do retangulo do rosto
                regiao_olhos       = imagem[y:y +a, x:x + l]
                regiao_cinza_olhos = cv2.cvtColor(regiao_olhos, cv2.COLOR_BGR2GRAY)
                olhos_detectados   = self.classificador_olhos.detectMultiScale(regiao_cinza_olhos)

                # retangulos nos olhos
                for (ox, oy, ol, oa) in olhos_detectados:
                    cv2.rectangle(regiao_olhos, (ox, oy), (ox+ol, oy+oa), (0, 255,0), 2)

                    # captura imagem ao pressionar tecla 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):

                        print(np.average(imagem_cinza))

                        # verifica claridade da imagem, ( padrão 110 )
                        if np.average(imagem_cinza) > 60:
                            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (self.largura_foto, self.altura_foto))
                            cv2.imwrite('fotos/pessoa.'+str(id_pessoa)+'.'+str(self.amostra)+'.jpg', imagem_face)
                            print('foto '+str(self.amostra)+' capturada!')
                            self.amostra += 1

            cv2.imshow("face", imagem)
            cv2.waitKey(1)
            if (self.amostra >= self.numero_amostras + 1):
                break

        camera.release()
        cv2.destroyAllWindows()