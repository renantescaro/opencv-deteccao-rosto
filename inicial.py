import cv2

# parametro 0 para webcam
camera = cv2.VideoCapture('http://192.168.0.200:8080/?action=stream')
classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while (True):
    conectado, imagem = camera.read()
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(100, 100))

    # retangulos em faces detectadas
    for (x, y, l, a) in faces_detectadas:
        cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)

    cv2.imshow("teste", imagem)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()