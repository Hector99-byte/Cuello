import cv2 as cv 
from math import atan2, cos, sin, sqrt, pi
import numpy as np  
import argparse
from objeto import Objeto
from medida import Medida

def dibujarNombre(imagen, puntoCentral, a, p, de, excent):
	#Definicion de los ojetos
	conector = Objeto("Conector", (1600, 200), (170, 15), (44, 5), (0.65, 0.2))
	bateria = Objeto("Bateria", (2800, 500), (375, 125), (60, 5), (0.53, 0.3))
	vasito = Objeto("Vasito", (14500, 1500), (570, 80), (130, 10), (0.64, 0.4))
	lata = Objeto("lata", (4100, 1600), (750, 230), (65, 25), (0.6, 0.2))

	conector.calculoValoresUmbral(a, p, de, excent)
	bateria.calculoValoresUmbral(a, p, de, excent)
	vasito.calculoValoresUmbral(a, p, de, excent)
	lata.calculoValoresUmbral(a, p, de, excent)

	if conector.isObjeto():
		cv.putText(imagen, conector.nombre, puntoCentral, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 180), 1)
	elif bateria.isObjeto():
		cv.putText(imagen, bateria.nombre, puntoCentral, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 180), 1)
	elif vasito.isObjeto():
		cv.putText(imagen, vasito.nombre, puntoCentral, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 180), 1)
	elif lata.isObjeto():
		cv.putText(imagen, lata.nombre, puntoCentral, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 180), 1)
	else:
		cv.putText(imagen, "Unknown", puntoCentral, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 180), 1)

	"""
	determinarObjeto(imagen, conector, puntoCentral)
	determinarObjeto(imagen, bateria, puntoCentral)
	determinarObjeto(imagen, vasito, puntoCentral)
	determinarObjeto(imagen, lata, puntoCentral)
"""
	cv.namedWindow("Etiquetado de objetos", cv.WINDOW_AUTOSIZE)
	cv.imshow("Etiquetado de objetos", imagen)
	
def determinarObjeto(imagen, objeto, puntoCentral):
	#print("Llego al etiquetado")
	if objeto.isObjeto():
		cv.putText(imagen, objeto.nombre, puntoCentral, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 180), 1)

def drawAxis(img, p_, q_, colour, scale):
	p = list(p_)
	q = list(q_)

	angle = atan2(p[1] - q[1], p[0] - q[0])
	hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

	q[0] = p[0] - scale * hypotenuse * cos(angle)
	q[1] = p[1] - scale * hypotenuse * sin(angle)
	cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
#	cv.line(img, (int(p[0])), (int(p[1])), (int(q[0])), (int(q[1])), colour, 1, cv.LINE_AA)


	p[0] = q[0] + 9 * cos(angle + pi / 4)
	p[1] = q[1] + 9 * sin(angle + pi / 4)
	cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
	#cv.line(img, (int(p[0])), (int(p[1])), (int(q[0])), (int(q[1])), colour, 1, cv.LINE_AA)
	

	p[0] = q[0] + 9 * cos(angle - pi / 4)
	p[1] = q[1] + 9 * sin(angle - pi / 4)
	cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
	#cv.line(img, (int(p[0])), (int(p[1])), (int(q[0])), (int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(pts, img, area, perimetro, de, extent):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
        
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    dibujarNombre(img, cntr, area, perimetro, de, extent)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle

def corregirBrillo(imagen, _escalar):
	escalar = list(_escalar)
	colorAzul = cv.add(imagen[:, :, 0], escalar[0])
	colorVerde = cv.add(imagen[:, :, 1], escalar[1])
	colorRojo = cv.add(imagen[:, :, 2], escalar[2])
	imagen[:, :, 0] = colorAzul
	imagen[:, :, 1] = colorVerde
	imagen[:, :, 2] = colorRojo

captura = cv.VideoCapture(1)

if not captura.isOpened():
	print("No se pudo conectar con la camara")
	exit()

"""
cv.namedWindow('algoritmo', cv.WINDOW_AUTOSIZE)
cv.namedWindow('binarizacion', cv.WINDOW_AUTOSIZE)
cv.namedWindow('sensor', cv.WINDOW_AUTOSIZE)
cv.namedWindow('Corregido brillo y constraste', cv.WINDOW_AUTOSIZE)
cv.namedWindow('Componentes conectados', cv.WINDOW_AUTOSIZE)
"""

tmp_nombre = 0


while True:
	ret, imagenNativa = captura.read()
	imagenCorrecionConstraste = imagenNativa.copy()
	#imagenMejorConstraste
	cv.imshow('Imagen normal', imagenNativa)

	if not ret:
		print("Ya no se obtuvo imagen")
		break

	#cv.imshow('Imagen nativa', imagenNativa)

	"""Convirtiendo primero la matriz y despues usando el factor de 
		escalacion
	imagenCorrecionConstraste = np.float32(imagenNativa)
	imagenCorrecionConstraste *= 4"""
	
	#Mejora el contraste
	cv.convertScaleAbs(imagenNativa, imagenCorrecionConstraste, 4, 0 )
	imagenCorregida = imagenCorrecionConstraste.copy()
	corregirBrillo(imagenCorregida, (-175, -180, -150))

	cv.imshow('Imagen corregida', imagenCorregida)
	cv.imshow('Imagen nativa', imagenNativa)
	#imagenCorregidaGray = imagenCorregida.copy()

	distorMat = np.array([-0.0688081, 0.101627, -0.000487848, -0.00172756, -0.0388046])
	cameraMatrix = np.array([[893.035, 0, 623.697], [0, 895.748, 526.612], [0, 0, 1]])

	imagenCorregidaGray = cv.cvtColor(imagenCorregida, cv.COLOR_BGR2GRAY)
	imagenCorregidaGray = np.uint8(imagenCorregidaGray)

	uPhoto = imagenCorregidaGray.copy()

	k1 = distorMat[0]
	k2 = distorMat[1]
	p1 = distorMat[2]
	p2 = distorMat[3]
	k3 = distorMat[4]
	fx = cameraMatrix[0, 0]
	cx = cameraMatrix[0, 2]
	fy = cameraMatrix[1, 1]
	cy = cameraMatrix[1, 2]
	z = 1

	"""for i in range(imagenCorregidaGray.shape[1]):
		for j in range(imagenCorregidaGray.shape[0]):
			x = (i - cx) / fx
			y = (j - cy) / fy
			r2 = x * x + y * y

			dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
			dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
			scale = (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

			xBis = x * scale + dx
			yBis = y * scale + dy 

			xCorr = xBis * fx + cx
			yCorr = yBis * fy + cy 

			if xCorr >= 0 and xCorr < uPhoto.shape[1] and yCorr >= 0 and yCorr < uPhoto.shape[0]:
				uPhoto[int(yCorr), int(xCorr)] = imagenCorregidaGray[j, i]"""

	_ ,thres = cv.threshold(uPhoto, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)

	cv.imshow("Binarizacion", thres)
	labelImage = np.empty(thres.shape, dtype = np.int32())

	nlabels, retVal = cv.connectedComponents(thres, labelImage, 8, cv.CV_32S)
	colors = list()
	colors.insert(0, (0, 0, 0))

	
	for label in range(1, nlabels):
		colors.insert(label, (120 * label, 50 * label, 200 * label))

	coloreada = np.empty(thres.shape, np.uint8())
	

	"""
	for r in range(coloreada.shape[0]):
		for c in range(coloreada.shape[1]):
			label = labelImage[r, c]
			pixel = coloreada[r, c]
			pixel = colors[label]"""

	contours, hierarchy = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	#contours, hierarchy = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	drawing = np.zeros((imagenCorregida.shape[0], imagenCorregida.shape[1]), np.uint8)
	color = np.array((255, 255, 255), np.uint8)

	for i in range(len(contours)):
		perimetro = cv.arcLength(contours[i], True)
		area = cv.contourArea(contours[i])

		if area > 200:
			#cv.drawContours(drawing, contours, i, color, 3)
			boundingRect = cv.boundingRect(contours[i])
			print(boundingRect)
			de = sqrt(4 * area / 3.1416)
			rect_area = boundingRect[2] * boundingRect[3]
			extent = area / rect_area

			if len(contours[i]) > 5:
				minEllipse = cv.fitEllipse(contours[i])
			
			print('area: ', area)
			print('perimetro: ', perimetro)
			print('de: ', de)
			print('extent', extent)

			getOrientation(contours[i], imagenCorregida, area, perimetro, de, extent)


	if cv.waitKey(1) == ord('q'):
		break

captura.release()
cv.destroyAllWindows()
#dibujarNombre(10, 20, 50, 80, 30, 50)
