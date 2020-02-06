import cv2 as cv

class Camara:

    def __init__(self, numeroCamara):
        self.captura = cv.VideoCapture(numeroCamara)

    def isCamaraConnected(self):
        if not self.captura.isOpened():
            print("No se pudo conectar con la camara")
            return False
        else:
            return True

    def tomarImagen(self):
        if self.isCamaraConnected():
            self.imagenOriginal = self.captura.read()
            self.corregirConstraste()
            self.corregirBrillo()

            return self.imagenCorregida

        return None

    def corregirConstraste(self):
        self.imagenCorregida = cv.convertScaleAbs(self.imagenOriginal, self.imagenCorregida, 4, 0 )

    def corregirBrillo(self):
        self.imagenCorregida[:, :, 0] = cv.add(self.imagenOriginal[:, :, 0], -175)     #Canal azul
        self.imagenCorregida[:, :, 1] = cv.add(self.imagenOriginal[:, :, 1], -180)     #Canal  verde
        self.imagenCorregida[:, :, 2] = cv.add(self.imagenOriginal[:, :, 2], -150)     #Canal rojo

    def getImagenOriginal(self):
        return self.imagenOriginal

    def getImagenEnEscalaDeGrises(self):
        if self.imagenCorregida != None:
            return cv.cvtColor(self.imagenCorregida, cv.COLOR_BGR2GRAY)

        return None

    def regrasarImagenes(self):
        #Regresa la imagen corregida(Brillo, contraste), imagen original, imagen en escala de grises
        return self.tomarImagen(), self.getImagenOriginal(), self.getImagenEnEscalaDeGrises()