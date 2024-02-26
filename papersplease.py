import cv2
import numpy as np

#importar tensorflow
import tensorflow as tf

#criar variável model para receber o modelo treinado. 
model = tf.keras.models.load_model("keras_model.h5")

video = cv2.VideoCapture(1)

while True:

    check,frame = video.read()

    # Altere o dado de entrada:
    # 1. Redimensione a imagem
    img = cv2.resize(frame, (224,224))

    

    # 2. Converta a imagem em um array Numpy e aumente a dimensão para o funcionamento correto das imagens treinadas
    test_image = np.array(img,dtype=np.float32)
    test_image = np.expand_dims(test_image,axis=0)
    # 3. Normalize a imagem (ajuda no treinamento eficaz da máquina, especialmente em redes neurais) (rapidez no processamento de imagens)
    normalised_image = test_image/255.0

    # Preveja o resultado
    prediction = model.predict(normalised_image.os)

   

    print("Jogada: ", prediction)
        
    cv2.imshow("pedrapapeltesoura.exe",frame)
            
    key = cv2.waitKey(1)

    if key == 32:
        print("Fechando")
        break

video.release()
