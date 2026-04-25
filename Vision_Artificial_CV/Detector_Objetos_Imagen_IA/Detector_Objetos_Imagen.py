# El siguiente software sirve para encontrar objetos en una imagen
# el software muestra cada objeto y que tipo de objeto es
# Usando la librerias cv2 y sus datos
# En el terminal de comandos importar librerias con el comando PIP 
# pip install numpy opencv-python matplotlib

import cv2
import matplotlib.pyplot as plt
import cvlib as cv 
#from tensorflow import input
from cvlib.object_detection import draw_bbox

im = cv2.imread('manzana.jpg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()
