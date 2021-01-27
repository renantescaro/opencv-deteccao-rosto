from captura_faces import CapturaFaces
from treinamento import Treinamento

captura_faces = CapturaFaces()
# treinamento = Treinamento()

# captura de imagens
id_pessoa = input('Id da pessoa: ')
captura_faces.iniciar(int(id_pessoa))

# treinamento ( no minimo 2 pessoas com faces capturadas )
# treinamento.iniciar()