# Librerias del generador de directorios
import zipfile
import os
import shutil

# Librerias de la Red
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
#from keras.applications import ResNet50
from keras import models, layers
from keras.callbacks import ModelCheckpoint
from keras import optimizers

zipFilenames = ["frog.zip","spider.zip","snake.zip","Wild_Anim_Dataset.zip"]

for filename in zipFilenames:
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall()
    zip_ref.close()

# Creamos las variables para almacenar los nombres de los directorios
# de prueba, validación y entrenamiento, usamos un arreglo con los nombres
# de los animales, para usar como base de procesamiento, y creación de categorías

animals = ["frog","spider","snake","bear","elephant","leopard","lion","wolf"]
train_anim_dirs = []
validation_anim_dirs = []
test_anim_dirs = []

# Declaramos el nombre de la ruta de origen, y de la ruta de destino
original_dataset_dir = "Wild_Anim_Dataset/Wild_Anim_5000_images/"
base_dir = "Animals"

# Alimentamos la ruta de destino, que va a crear, el siguiente árbol de
# directorios:
# ---------------> Animals
#                    |----------> Test
#                           |----------> Anim1
#                           |----------> Anim2
#                           :
#                           .
#                           |----------> AnimN
#                    |----------> Train
#                           |----------> Anim1
#                           |----------> Anim2
#                           :
#                           .
#                           |----------> AnimN
#                    |----------> Validation
#                           |----------> Anim1
#                           |----------> Anim2
#                           :
#                           .
#                           |----------> AnimN
# Aquí creamos las 3 carpetas principales del árbol
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Ahora alimentamos las 3 carpetas con los subdiretorios de animales
# nótese
for animal in animals:
    train_anim_dirs.append(os.path.join(train_dir, animal)) #[0].capitalize()+animal[1:]))
    validation_anim_dirs.append(os.path.join(validation_dir, animal))
    test_anim_dirs.append(os.path.join(test_dir,animal))

# Ahora creamos la capeta principal o contenedora del árbol (la raíz)
os.mkdir(base_dir)

# Aquí creamos las 3 primeras ramas del árbol de directorios
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

# Alimentamos el árbol con las carpetas donde almacenaremos
# las imagenes por animal.
for i in range(len(train_anim_dirs)):
    os.mkdir(train_anim_dirs[i])
    os.mkdir(validation_anim_dirs[i])
    os.mkdir(test_anim_dirs[i])

fnames =[] #[fnames'{}{}.jpg'.format(j,i) for j in Animals for i in range(1000)]

# Aquí creamos los nombres que tendrán las imágenes, el primer for
# es para crear los nombres de las imágenes que descargamos de internet
# dado que sólo son 100 en las primeras 3 clases del array animals
# es decir, elementos 0,1,2, barremos sólo del 0 al 99 en el for anidado
for animal in animals[:3]:
    tempFnames = []
    for j in range(100):
        tempFnames.append('{}{}.jpg'.format(animal,j))
    fnames.append(tempFnames)
# Para el resto de elementos del array Animals, barremos del 0 al 999
# dado que tenemos 1000 imágenes por carpeta.
for animal in animals[3:]:
    tempFnames = []
    for j in range(1000):
        tempFnames.append('{}{}.jpg'.format(animal,j))
    fnames.append(tempFnames)

"""
Depurador del código
for i in range(len(train_anim_dirs)):
    print(train_anim_dirs[i])
    print(validation_anim_dirs[i])
    print(test_anim_dirs[i])
    print(animals[i])

for i in fnames:
    print(i[0])
"""

# Aquí debemos comenzar a mover los archivos a su carpetas de destino
for i in range(len(animals)):
    k = 0
    if i >= 3:
        subfolder_dir = os.path.join(original_dataset_dir,
        animals[i][0].capitalize()+animals[i][1:])
        original_fnames = os.listdir(subfolder_dir)
    elif i >= 0 and i <= 2:
         subfolder_dir = animals[i]
         original_fnames = os.listdir(subfolder_dir)

    Q = len(original_fnames)
    Q1 = int(Q*.70)
    Q2 = int(Q*.85)
    # Train
    print(i)
    for j in range(0,Q1):
        src = os.path.join(subfolder_dir, original_fnames[j])
        dst = os.path.join(train_anim_dirs[i], fnames[i][k])
        shutil.copyfile(src, dst)
        k += 1
    # Validation
    for j in range(Q1,Q2):
        src = os.path.join(subfolder_dir, original_fnames[i])
        dst = os.path.join(validation_anim_dirs[i], fnames[i][k])
        shutil.copyfile(src, dst)
        k += 1
    # Test
    for j in range(Q2,Q):
        src = os.path.join(subfolder_dir, original_fnames[i])
        dst = os.path.join(test_anim_dirs[i], fnames[i][k])
        shutil.copyfile(src, dst)
        k += 1

# Cuidado poner las direcciones correctas de las carpetas
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_anim_dirs[0],
                                   target_size=(200, 200),
                                   batch_size=20,
                                   class_mode="categorical")
val_generator = datagen.flow_from_directory(validation_anim_dirs[0],
                                   target_size=(200, 200),
                                   batch_size=20,
                                   class_mode="categorical")
test_generator = datagen.flow_from_directory(test_anim_dirs[0],
                                   target_size=(200, 200),
                                   batch_size=20,
                                   class_mode="categorical")

resnet50_base = ResNet50(weights='imagenet',include_top=False, pooling="avg",input_shape=(200,200,3))
resnet50olp = models.Sequential()
resnet50olp.add(resnet50_base)
resnet50olp.add(layers.Dense(8, activation="sigmoid"))
resnet50olp.summary()

resnet50olp.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])
checkpointer = ModelCheckpoint(filepath="Animals8.h5", monitor='val_acc', verbose=0,
                               save_best_only=True, mode='max', period=1)
h = resnet50olp.fit_generator(train_generator,steps_per_epoch=3659//20,epochs=10,
                              validation_data=val_generator,validation_steps=1833//20,
                              callbacks=[checkpointer])

epoch_max = np.argmax(h.history['val_acc'])
plt.plot(h.history['acc'], label='Training acc')
plt.plot(h.history['val_acc'], label='Validation acc')
plt.legend(loc='lower right')
plt.plot(epoch_max, h.history['val_acc'][epoch_max],'*')
plt.title('Accuracy')
plt.show()
