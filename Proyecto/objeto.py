import cv2
from medida import Medida
class Objeto:

	def __init__(self, nombre, medicion1, medicion2, medicion3, medicion4):
		self.nombre = nombre
		self.setMedicion1(medicion1)
		self.setMedicion2(medicion2)
		self.setMedicion3(medicion3)
		self.setMedicion4(medicion4)

	def setMedicion1(self, _medicion1):
		self.medicion1 = Medida()
		medicion1 = list(_medicion1)
		self.medicion1.setValorEsperado(medicion1[0])
		self.medicion1.setUmbral(medicion1[1])

	def setMedicion2(self, _medicion2):
		self.medicion2 = Medida()
		medicion2 = list(_medicion2)
		self.medicion2.setValorEsperado(medicion2[0])
		self.medicion2.setUmbral(medicion2[1])

	def setMedicion3(self, _medicion3):
		self.medicion3 = Medida()
		medicion3 = list(_medicion3)
		self.medicion3.setValorEsperado(medicion3[0])
		self.medicion3.setUmbral(medicion3[1])

	def setMedicion4(self, _medicion4):
		self.medicion4 = Medida()
		medicion4 = list(_medicion4)
		self.medicion4.setValorEsperado(medicion4[0])
		self.medicion4.setUmbral(medicion4[1])

	def calculoValoresUmbral(self, a, p, de, excent):
		self.medicion1.setMedicionObtenida(self.medicion1.valorEsperado - a)
		self.medicion2.setMedicionObtenida(self.medicion2.valorEsperado - p)
		self.medicion3.setMedicionObtenida(self.medicion3.valorEsperado - de)
		self.medicion4.setMedicionObtenida(self.medicion4.valorEsperado - excent)
		self.ajustarValores()

	def ajustarValores(self):
		if self.medicion1.medicionObtenida < 0:
			self.medicion1.medicionObtenida *= -1
		if self.medicion2.medicionObtenida < 0:
			self.medicion2.medicionObtenida *= -1
		if self.medicion3.medicionObtenida < 0:
			self.medicion3.medicionObtenida *= -1
		if self.medicion4.medicionObtenida < 0:
			self.medicion4.medicionObtenida *= -1

	def isObjeto(self):
		if self.medicion1.medicionObtenida <= self.medicion1.umbral and self.medicion2.medicionObtenida <= self.medicion2.umbral and self.medicion3.medicionObtenida <= self.medicion3.umbral and self.medicion4.medicionObtenida <= self.medicion4.umbral:
		#if self.medicion1.medicionObtenida <= self.medicion1.umbral and self.medicion2.medicionObtenida <= self.medicion2.umbral and self.medicion3.medicionObtenida <= self.medicion3.umbral:
			return True

		return False