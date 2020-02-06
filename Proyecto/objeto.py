from recordtype import recordtype

class Objeto:

	Mediciones = recordtype('mediciones', 'valor umbral umbralCalculado')

	def __init__(self,  valoresArea, valoresPerimetro, valoresDiametroEquivalente, valoresExtent):
		self.area = self.Mediciones(valoresArea[0], valoresArea[1], 0)
		self.perimetro = self.Mediciones(valoresPerimetro[0], valoresPerimetro[1], 0)
		self.diametroExterno = self.Mediciones(valoresDiametroEquivalente[0], valoresDiametroEquivalente[1], 0)
		self.extent = self.Mediciones(valoresExtent[0], valoresExtent[1], 0)

	def calcularUmbral(self, area, perimetro, diametroExterno, extent):

		self.area.umbralCalculado = self.area.valor - area
		self.perimetro.umbralCalculado = self.perimetro.valor - perimetro
		self.diametroExterno.umbralCalculado = self.diametroExterno.valor - diametroExterno
		self.extent.umbralCalculado = self.extent.valor - extent
		self.convertirAValoresAbsolutos()

	def convertirAValoresAbsolutos(self):
		if self.area.umbralCalculado < 0:
			self.area.umbralCalculado *= -1
		if self.perimetro.umbralCalculado < 0:
			self.perimetro.umbralCalculado *= -1
		if self.diametroExterno.umbralCalculado < 0:
			self.diametroExterno.umbralCalculado *= -1
		if self.extent.umbralCalculado < 0:
			self.extent.umbralCalculado *= -1

	def isObjeto(self):
		if self.area.umbralCalculado <= self.area.umbral and self.perimetro.umbralCalculado <= self.perimetro.umbral and self.diametroExterno.umbralCalculado <= self.diametroExterno.umbral and self.extent.umbralCalculado <= self.extent.umbral:
		#if self.medicion1.medicionObtenida <= self.medicion1.umbral and self.medicion2.medicionObtenida <= self.medicion2.umbral and self.medicion3.medicionObtenida <= self.medicion3.umbral:
			return True

		return False

