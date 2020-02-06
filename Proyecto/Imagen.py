import cv2 as cv

class Imagen:

    def __init__(self, imagenOriginal, imagenCorregida):
        self.imageOriginal = imagenOriginal
        self.imagenCorregida = imagenCorregida
        self.imagenEscalaGrises = cv.cvtColor(imagenCorregida, cv.COLOR_BGR2GRAY)

print(cv.__version__)
