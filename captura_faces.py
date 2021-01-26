import cv2

# parametro 0 para webcam integrada / usb
camera = cv2.VideoCapture('http://192.168.0.200:8080/?action=stream')
classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
amostra = 1
numero_amostras = 25
largura_foto, altura_foto = 220, 220

# identificador único da pessoa
id_pessoa = input('Digite o identificador da pessoa: ')

while (True):
    conectado, imagem = camera.read()
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150, 150))

    # retangulos em faces detectadas
    for (x, y, l, a) in faces_detectadas:
        cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)

        # captura imagem ao pressionar tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura_foto, altura_foto))
            cv2.imwrite('fotos/pessoa.'+str(id_pessoa)+'.'+str(amostra)+'.jpg', imagem_face)
            print('foto '+str(amostra)+' capturada!')
            amostra += 1

    cv2.imshow("teste", imagem)
    cv2.waitKey(1)
    if (amostra >= numero_amostras + 1):
        break

camera.release()
cv2.destroyAllWindows()