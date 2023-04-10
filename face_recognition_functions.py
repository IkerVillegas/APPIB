import cv2
import numpy as np
from scipy.spatial import distance
import tensorflow.keras.backend as K
import tensorflow.keras.models as Models
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os
from keras.utils import np_utils

#Detector de caras basado en redes convolucionales 2D
detector_dnn = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

#CARGAMOS EL MODELO DE RECONOCIMIENTO FACIAL basado en Resnet-50 y entrenado con VGG-Face
model_file = 'resnet50.h5'
model = Models.load_model(model_file)
last_layer = model.get_layer('avg_pool').output
feature_layer = Flatten(name = 'flatten')(last_layer)
feature_extractor = Models.Model(model.input, feature_layer)


def extract_faces(file_img):
    #Función que a partir de una imagen, detecta y recorta la cara
    img =  cv2.imread(file_img, cv2.IMREAD_UNCHANGED)
    centro=[int(img.shape[1]/2),int(img.shape[0]/2)]
    (h, w) = img.shape[:2]
    inputBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1, (300, 300), (104, 177, 123))
    detector_dnn.setInput(inputBlob)
    detections = detector_dnn.forward()
    list_box=[] 
    distancia=[]
    #Detectar la cara y recortarla
    if detections.shape[2]<=0:
             print("Cara no detectada")
    else:
             for i in range(0, detections.shape[2]):
                     prediction_score = detections[0, 0, i, 2]
                     if prediction_score >0.8:
                      
                       box1 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                       iz,arri,dere,abajo=box1.astype("int")
                       if iz<0 or iz>img.shape[1]:
                           iz=0
         
                       if arri<0 or arri>img.shape[0]:
                          arri=0
             
                       if dere> img.shape[1]:
                          dere= img.shape[1]
                   
                       if abajo> img.shape[0]:
                          abajo= img.shape[0]
                       list_box.append([iz,arri,dere,abajo])
                       centro1=[int((dere+iz)/2),int((abajo+arri)/2)]
                       distancia.append(distance.euclidean(centro, centro1))
                       
             if len(distancia)>0:          
                 box=list_box[np.argmin(distancia)]
                 imagen_copia=img
                 imagen_copia=imagen_copia[box[1]:box[3],box[0]:box[2]]
                 list_box=[] 
                 distancia=[]
             else:
                 print("No detectado")
                 imagen_copia=[]
    return imagen_copia

def preprocess_input(x, data_format=None, version=2):
    #Función de pre-procesado de la imagen antes de ser introducida en el modelo resnet-50 
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def generate_embedding(img):
    #Función que genera un embedding a partir de una cara y el modelo entrenado
  
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
 
    #img = image.load_img(files)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img, version = 2)
    
    emb = feature_extractor.predict(img)
    emb_norm = preprocessing.normalize(emb, norm = 'l2', axis = 1, copy = True,
                                       return_norm = False)
    
    return emb_norm


def obtain_data_embedding(carpeta, n_identidades, n_archivos):
    ''' 
    Obtención de la lista de embeddings para un conjunto de archivos pertenecientes a un directorio local

    PARAMETROS
    ------------
    v_carpeta: lista de carpetas donde se encuentran los archivos a usar
    n_identidades: número de identidades a analizar
    n_archivos: número de imágenes por identidad 

    SALIDA
    ------------
    X: matriz de datos con los embeddings situados por filas
    y: lista con las etiquetas por género para cada embedding
    '''
    X=[] # lista de embedings
    y=[] # lista de etiquetas 
    
    #for carpeta in v_carpeta:

    aux = 0 #controlador para el número de identidades
    directorio = "4K_120\\" + carpeta #contruimos el directorio a recorrer

    # Recorremos de manera recursiva todos los archivos y carpetas en la ruta especificada
    for ruta, carpetas, archivos in os.walk(directorio):

        # Recorremos todos los archivos en la carpeta actual
        for archivo in archivos[0:n_archivos]:

            # Generamos la ruta completa del archivo
            ruta_archivo = os.path.join(ruta, archivo)
            img =  cv2.imread(ruta_archivo, cv2.IMREAD_UNCHANGED)

            #Extraemos los embeddings del archivo
            embeddings = generate_embedding(img)

            #Agregamos los embeddings a la lista 
            X.append(embeddings[0])

            #Agregamos la etiqueta a la lista
            if carpeta[0] == 'H':
                y.append(0)
            else:
                y.append(1)

            # Parar bucle 
            aux = aux +1
            if aux>=n_identidades:
                break
        else:
            continue
        break
    
    X = np.array(X)
    return X,y

def obtain_train_test(v_carpeta, n_identidades = 750, n_archivos = 1):
    ''' 
    Obtención de datos de entrenamiento y de test con sus etiquetas correspondientes

    PARAMETROS
    ------------
    v_carpeta: primer embedding
    n_identidades: número de identidades a tomar de los datos
    n_archivos: número de imágenes por identidad a considerar

    SALIDA
    ------------
    X_train: Matriz de datos para entrenamiento
    y_train: Vector con las etiquetas para los datos de entrenamiento
    X_test: Matriz de datos para test
    y_test: Vector con las etiquetas para los datos de test
    '''

    index_train = 500

    X_0, y_0 = obtain_data_embedding(v_carpeta[0], n_identidades, n_archivos)
    X_1, y_1 = obtain_data_embedding(v_carpeta[1], n_identidades, n_archivos)

    ## DATOS DE ENTRENAMIENTO
    # Seleccionamos una parte de los datos de embeding 
    X = list(X_0[0:index_train]) + list(X_1[0:index_train]) 
    y = y_0[0:index_train] + y_1[0:index_train]
    # Los arreglamos para ser argumentos adecuados para el fit
    X_train = np.array(X).astype(float)              
    y_train = np_utils.to_categorical(np.asarray(y))

    ## DATOS DE TEST
    # Seleccionamos una parte de los datos de embeding 
    index_test = index_train # 30% de los datos
    X = list(X_0[index_test:]) + list(X_1[index_test:]) # asiáticos
    y = y_0[index_test:] + y_1[index_test:]
    # Los arreglamos para ser argumentos adecuados para el fit
    X_test = np.array(X).astype(float) 
    y_test = np_utils.to_categorical(np.asarray(y))

    return X_train, y_train, X_test, y_test

def similitud_embedding(x, y):
    ''' 
    Obtención de similaridad entre dos embeddings

    PARAMETROS
    ------------
    x: primer embedding
    y: segundo embedding

    SALIDA
    ------------
    sim: valor de similaridad entre los dos embeddings
    '''
    embedding_x = np.asarray(x)
    f1 = np.squeeze(embedding_x) #función que convierte el embedding en un elemento unidimensional
    embedding_y = np.asarray(y)
    f2 = np.squeeze(embedding_y)

    # Similaridad basada en el producto vectorial. Valores altos significan alta similaridad
    sim=np.dot(f1, f2.T)

    return sim

def get_scores(x):
    ''' 
    Obtención de dos listas con las puntuaciones de similaridad clasificadas según sean genuinos o impostores

    PARAMETROS
    ------------
    x: lista de embeddings

    SALIDA
    ------------
    score_genuino: lista con las puntuaciones clasificadas como genuinas
    score_impostor: lista con las puntuaciones clasificadas como impostoras
    '''
    score_genuino = [] #lista para almacenar las puntuaciones genuinas
    score_impostor = [] #lista para almacenar las puntuaciones impostoras

    length = len(x) #longitud de la lista de entrada
    index_jpg = [i for i in range(length)] #lista para controlar que no se repitan comparaciones

    for i in range(length):
        
        for j in index_jpg:
            if i!=j:
                sim = similitud_embedding(x[i],x[j]) #calculo de la similaridad

                #clasificación a genuino o impostor en base a si supera el valor 0.5
                if sim>=0.50:
                    score_genuino.append(sim)
                else:
                    score_impostor.append(sim)
        
        #eliminamos el índice para evitar que se repita la comparación en el caso inverso 
        index_jpg.remove(i)

    return score_genuino, score_impostor
    

def curva_FAR_FRR(score_genuino, score_impostor):
    ''' 
    Salida por pantalla de las curvas FAR y FRR.

    PARAMETROS
    ------------
    score_genuino: lista con las puntuaciones clasificadas como genuinas
    score_impostor: lista con las puntuaciones clasificadas como impostoras
    '''

    sorted_genuino = sorted(score_genuino)
    sorted_impostor = sorted(score_impostor)
    
    ax = pd.DataFrame(sorted_genuino).plot(kind='density')
    pd.DataFrame(sorted_impostor).plot(ax=ax,kind='density')
    
    #ax.axvline(0.434, color='g', linestyle='--')
    ax.legend(["Genuines Distribution","Impostors Distribution"])
    #ax.text(0.38,-0.3,"FRR")
    #ax.text(0.45,-0.3,"FAR")
    plt.xlabel("Matching score value")

    plt.show()

def representacion_TSNE(XA, XB, XN, n_components = 2, perplexity = 40):
    """ 
    La función TSNE coge vectores y los reduce a dimensión = n_components, en este caso =2. 
    Los valores del vector nuevo [x0,x1] se sacan en proporción a la distancia con el resto de sus vecinos
    No tiene atributo .transform(), por lo que pese a que se puede entrenar para una sola clase con .fit(),
    no se puede evaluar otras clases para ese entrenamiento. Solo se queda hacer .fit_transform() a la vez.
    Se saca por pantalla cuatro gráficos en base a los datos obtenidos: uno de los datos masculinos, un segundo
    con los datos femeninos, un tercero con ambos sexos y un último igual que el anterior pero diferenciando
    sexos además de etnias.

    PARAMETROS
    ------------
    XA: datos asiáticos
    XB: datos caucásicos
    XN: datos negros
    n_components: número de componentes para realizar la reducción de dimensionalidad
    perplexity: número de vecinos a considerar
    '''
    """
    tsne_model = TSNE(n_components=n_components, learning_rate='auto',init='random',perplexity=perplexity)

    # Hombres y mujeres
    xH_embedded = tsne_model.fit_transform(np.concatenate((XA[0:499],XB[0:499],XN[0:499])))
    xM_embedded = tsne_model.fit_transform(np.concatenate((XA[500:],XB[500:],XN[500:])))
    # Ambos géneros
    x_embedded = tsne_model.fit_transform(np.concatenate((XA,XB,XN)))


    ## PLOTS ##
    figH = plt.figure()
    axH=figH.add_subplot()
    axH.scatter(xH_embedded[0:499,0], xH_embedded[0:499,1],label='Asiáticos',color='blue',alpha=0.8)
    axH.scatter(xH_embedded[500:999,0], xH_embedded[500:999,1],label='Caucásicos',color='red',alpha=0.8)
    axH.scatter(xH_embedded[1000:,0], xH_embedded[1000:,1],label='Negros',color='green',alpha=0.8)
    axH.set_title('Similitud entre razas para hombres')
    axH.legend()

    figM = plt.figure()
    axM=figM.add_subplot()
    axM.scatter(xM_embedded[0:499,0], xM_embedded[0:499,1],label='Asiáticas',color='blue',alpha=0.8)
    axM.scatter(xM_embedded[500:999,0], xM_embedded[500:999,1],label='Caucásicas',color='red',alpha=0.8)
    axM.scatter(xM_embedded[1000:,0], xM_embedded[1000:,1],label='Negras',color='green',alpha=0.8)
    axM.set_title('Similitud entre razas para mujeres')
    axM.legend()

    fig = plt.figure()
    ax=fig.add_subplot()
    ax.scatter(x_embedded[0:999,0], x_embedded[0:999,1],label='Asiáticos',color='blue',alpha=0.8)
    ax.scatter(x_embedded[1000:1999,0], x_embedded[1000:1999,1],label='Caucásicos',color='red',alpha=0.8)
    ax.scatter(x_embedded[2000:,0], x_embedded[2000:,1],label='Negros',color='green',alpha=0.8)
    ax.set_title('Similitud entre razas')
    ax.legend()

    fig = plt.figure()
    ax=fig.add_subplot()
    ax.scatter(x_embedded[0:499,0], x_embedded[0:499,1],label='Asiáticos',color='blue',alpha=0.8)
    ax.scatter(x_embedded[500:999,0], x_embedded[500:999,1],label='Asiáticas',color='cornflowerblue',alpha=0.8)
    ax.scatter(x_embedded[1000:1499,0], x_embedded[1000:1499,1],label='Caucásicos',color='red',alpha=0.8)
    ax.scatter(x_embedded[1500:1999,0], x_embedded[1500:1999,1],label='Caucásicas',color='tomato',alpha=0.8)
    ax.scatter(x_embedded[2000:2499,0], x_embedded[2000:2499,1],label='Negros',color='green',alpha=0.8)
    ax.scatter(x_embedded[2500:,0], x_embedded[2500:,1],label='Negras',color='springgreen',alpha=0.8)
    ax.set_title('Similitud entre razas')
    ax.legend()

    plt.show()