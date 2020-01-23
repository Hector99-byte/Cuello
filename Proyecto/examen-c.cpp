#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cstdlib>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;

//**********************************************ESPECIFICACION DE METODOS NECESARIOS**************************************************************
//************************drawAxis:Dibuja los dos egien vectors del PCA en linea******************************************************************
//************************dibujaNombre:Pone etiqueta de texto sobre cada objeto en linea**********************************************************
//************************getOrientation:Calcula el angulo, punto central y la orientacion del objeto en linea************************************

//**********************************************DECLARACION DE METODOS****************************************************************************
void drawAxis(Mat&, Point, Point, Scalar, const float);
void dibujaNombre(Mat&, Point, double a, double p, double de, double extent);
double getOrientation(const vector<Point> &, Mat&, double a, double p, double de, double extent);

//**********************************************METODO: dibujaNombre******************************************************************************
//***********************PARAMETROS DE ENTRADA: img=Imagen de captura en linea, centro= cntr, Punto central del objeto****************************
void dibujaNombre(Mat& img, Point centro, double a, double p, double de, double extent)
{
    	
	//Definicion de medias y umbrales parala funcion discriminante
	//CONECTOR
	double Conector_A=1600;
	double Umbral_C_A=200;
	double valor_C_A=0;

	double Conector_P=170;
	double Umbral_C_P=15;
	double valor_C_P=0;

	double Conector_DE=44;
	double Umbral_C_DE=5;
	double valor_C_DE=0;
	
	double Conector_E=0.65;
	double Umbral_C_E=0.2;
	double valor_C_E=0;
	
	//BATERIA
	double Bateria_A=2800;
	double Umbral_B_A=500;
	double valor_B_A=0;

	double Bateria_P=375;
	double Umbral_B_P=125;
	double valor_B_P=0;

	double Bateria_DE=60;
	double Umbral_B_DE=5;
	double valor_B_DE=0;
	
	double Bateria_E=0.53;
	double Umbral_B_E=0.3;
	double valor_B_E=0;

	//VASITO
	double Vasito_A=14500;
	double Umbral_V_A=1500;
	double valor_V_A=0;

	double Vasito_P=570;
	double Umbral_V_P=80;
	double valor_V_P=0;

	double Vasito_DE=130;
	double Umbral_V_DE=10;
	double valor_V_DE=0;
	
	double Vasito_E=0.64;
	double Umbral_V_E=0.4;
	double valor_V_E=0;

	//LATA
	double Lata_A=4100;
	double Umbral_L_A=1600;
	double valor_L_A=0;

	double Lata_P=750;
	double Umbral_L_P=230;
	double valor_L_P=0;

	double Lata_DE=65;
	double Umbral_L_DE=25;
	double valor_L_DE=0;
	
	double Lata_E=0.6;
	double Umbral_L_E=0.2;
	double valor_L_E=0;

	//CALCULO DE LOS VALORES UMBRAL DEL ELEMENTO CONECTOR
	valor_C_A=Conector_A-a;
	valor_C_P=Conector_P-p;
	valor_C_DE=Conector_DE-de;
	valor_C_E=Conector_E-extent;

	if(valor_C_A<0)
		valor_C_A*=-1;
	if(valor_C_P<0)
		valor_C_P*=-1;
	if(valor_C_DE<0)
		valor_C_DE*=-1;
	if(valor_C_E<0)
		valor_C_E*=-1;
	
	//CALCULO DE LOS VALORES UMBRAL DEL ELEMENTO BATERIA
	valor_B_A=Bateria_A-a;
	valor_B_P=Bateria_P-p;
	valor_B_DE=Bateria_DE-de;
	valor_B_E=Bateria_E-extent;

	if(valor_B_A<0)
		valor_B_A*=-1;
	if(valor_B_P<0)
		valor_B_P*=-1;
	if(valor_B_DE<0)
		valor_B_DE*=-1;
	if(valor_B_E<0)
		valor_B_E*=-1;

	//CALCULO DE LOS VALORES DE UMBRAL DEL ELEMENTO VASITO
	valor_V_A=Vasito_A-a;
	valor_V_P=Vasito_P-p;
	valor_V_DE=Vasito_DE-de;
	valor_V_E=Vasito_E-extent;

	if(valor_V_A<0)
		valor_V_A*=-1;
	if(valor_V_P<0)
		valor_V_P*=-1;
	if(valor_V_DE<0)
		valor_V_DE*=-1;
	if(valor_V_E<0)
		valor_V_E*=-1;

	//CALCULO DE LOS VALORES DE UMBRAL DEL ELEMENTO LATA
	valor_L_A=Lata_A-a;
	valor_L_P=Lata_P-p;
	valor_L_DE=Lata_DE-de;
	valor_L_E=Lata_E-extent;

	if(valor_L_A<0)
		valor_L_A*=-1;
	if(valor_L_P<0)
		valor_L_P*=-1;
	if(valor_L_DE<0)
		valor_L_DE*=-1;
	if(valor_L_E<0)
		valor_L_E*=-1;

	if((valor_C_A<=Umbral_C_A)&&(valor_C_P<=Umbral_C_P)&&(valor_C_DE<=Umbral_C_DE)&&(valor_C_E<=Umbral_C_E))
		{
		putText(img,"Conector",centro, FONT_HERSHEY_PLAIN,1,CV_RGB(0,0,180),1);
		}
	else if((valor_B_A<=Umbral_B_A)&&(valor_B_P<=Umbral_B_P)&&(valor_B_DE<=Umbral_B_DE)&&(valor_B_E<=Umbral_B_E))
		{
		putText(img,"Bateria",centro, FONT_HERSHEY_PLAIN,1,CV_RGB(0,0,180),1);
		}
	else if((valor_V_A<=Umbral_V_A)&&(valor_V_P<=Umbral_V_P)&&(valor_V_DE<=Umbral_V_DE)&&(valor_V_E<=Umbral_V_E))
		{
		putText(img,"Shot",centro, FONT_HERSHEY_PLAIN,1,CV_RGB(0,0,180),1);
		}
	else if((valor_L_A<=Umbral_L_A)&&(valor_L_P<=Umbral_L_P)&&(valor_L_DE<=Umbral_L_DE)&&(valor_L_E<=Umbral_L_E))
		{
		putText(img,"Lata",centro, FONT_HERSHEY_PLAIN,1,CV_RGB(0,0,180),1);
		}	
	else
		{
    		putText(img, "Unknown" , centro, FONT_HERSHEY_PLAIN, 1, CV_RGB(0,0,180), 1);
		}
    	namedWindow("Etiqueta Objeto & PCA",WINDOW_AUTOSIZE);
    	imshow("Etiqueta Objeto & PCA", img);
}

//**********************************************METODO: drawAxis**********************************************************************************
//***********************PARAMETROS DE ENTRADA: img=Imagen de captura en linea, p= punto donde termina el eigen vector p**************************
//***********************q= punto donde termina el eigen vector q, Scalar= dimenciones de los eigen vectors***************************************
//***********************colour= color y grosor con que se dibujan los eigen vectors, scale= escala de los eigen vectors respecto al objeto*******
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle;
    double hypotenuse;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, CV_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
}
double getOrientation(const vector<Point> &pts, Mat &img, double a, double p, double de, double extent)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2); 
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);
    dibujaNombre(img, cntr, a, p, de, extent);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;   
}

//*************************VARIABLES GLOBALES: prom_a= promedio del area, label2= indice de elementos del ciclo de captura en linea***************
//*************************sum_a= sumatoria de los valores calculados de area en linea************************************************************

int label2;
double caracteristicas [10000][6];// La cantidad de filas sera definido por 10000 datos para el algoritmo en linea
	
	double a;
	double p;
	double de;
	double extent;

		double A_prom=0;
		double sum_A=0;
		double sumatoria=0;

		double P_prom=0;
		double sum_P=0;
		double sumatoria1=0;

		double S_ejeM_prom=0;
		double sum_S_ejeM=0;
		double sumatoria2=0;

		double S_ejeMe_prom=0;
		double sum_S_ejeMe=0;
		double sumatoria3=0;

		double DE_prom=0;
		double sum_DE=0;
		double sumatoria4=0;

		double EXTENT_prom=0;
		double sum_EXTENT=0;
		double sumatoria5=0;

		int cuenta_datos=0;

//*************************FUNCION PRINCIPAL: main************************************************************************************************
int main(int argc,char** argv)
	{

	//Creamos el objeto cap de tipo VideoCapture para censar con el dispositivo los objetos presentados en el area de censado
	VideoCapture cap(1);
	if(!cap.isOpened())
		return -1;
	
	//VENTANAS DE INFORMACION RELEVANTE DEL PROCESO DE RECONOCIMIENTO
	namedWindow("algoritmo",WINDOW_AUTOSIZE);/*Muestra la imagen binarizada, con correccion de brillo, calibracion, solo falta nitidez, a y lo   		muestra con la inversion de blanco a negro y el recuadro que se extrae en base a la deteccion de bordes.*/
	namedWindow("binarizada",WINDOW_AUTOSIZE);//Muestra imagen con binarizacion y correcciones
	namedWindow("sensor",WINDOW_AUTOSIZE);//Nativa
	namedWindow("Corregido_brillo_contraste",WINDOW_AUTOSIZE);//Correccion brillo contraste
	//namedWindow("Componentes_conectados",WINDOW_AUTOSIZE);//Resultado de aplicar la funcion connected components
	namedWindow("Componentes Conectados",WINDOW_AUTOSIZE);
	//namedWindow("PCA",WINDOW_AUTOSIZE); 
	
	int tmp_nombre=0;
	for(;;)//Captura el objeto cap enun for infinito para que sea en linea, se terminara el ciclo al presionar una tecla
		{
		Mat frame, frame1, frame2;// En frame 2 se guardara la nativa
		cap>>frame2;
		imshow("sensor",frame2);//MOSTRAMOS LAS IMAGENES NATIVAS DEL CENSOR EN LINEA

		//Mejorar contraste y despues mejorar el brillo, el contraste aumento al doble y el brillo reduce en un 75 %
    	frame2.convertTo(frame1, -1, 4, 0);//En frame 1 se guardara la correccion de contraste, aunque me falta agregar el sharpness
		frame = frame1 + Scalar(-175, -180, -150);// En frame se guardara la correccion de brillo 
		
		Mat src= frame.clone(); //Creamos la imagen src a partir del frame ya corregido para usarlo en las funciones de PCA
		imshow("Corregido_brillo_contraste",frame);//MOSTRAMOS LAS IMAGENES CON CORRECCION DE BRILLO Y CONTRASTE DEL CENSOR EN LINEA 

		//Parametros intrisecos de distorcion y de calibracion de mi camara
		Mat distortMat = (Mat_<double>(1, 5) << -0.0688081, 0.101627, -0.000487848, -0.00172756, -0.0388046);
		Mat cameraMatrix = (Mat_<double>(3, 3) << 893.035, 0, 623.697, 0, 895.748, 526.612, 0, 0, 1);

		cvtColor(frame, frame, CV_BGR2GRAY);//Convertimos a escala de grises frame y sobreescribimos en frame	
		
		//Mat src= frame.clone();
		
		frame.convertTo(frame, CV_8UC1);
		Mat uPhoto = frame.clone(); // Se clona la imagen para que el algoritmo respete la imagen original

		double k1 = distortMat.at<double>(0, 0);  //Parametros de la matriz de distorcion que usara el algoritmo de correccion
		double k2 = distortMat.at<double>(0, 1);
		double p1 = distortMat.at<double>(0, 2);
		double p2 = distortMat.at<double>(0, 3);
		double k3 = distortMat.at<double>(0, 4);
		double fx = cameraMatrix.at<double>(0, 0);  //Parametros de la matriz de calibracion del censor que usara el algoritmo de correccion
		double cx = cameraMatrix.at<double>(0, 2);
		double fy = cameraMatrix.at<double>(1, 1);
		double cy = cameraMatrix.at<double>(1, 2);
		double z = 1.;

		for (int i = 0; i < frame.cols; i++) //Algoritmo de correccion: Incluye la calibracion y el ajuste de distorcion
			{
    			for (int j = 0; j < frame.rows; j++)
    				{
        			double x = (i - cx) / fx;
        			double y = (j - cy) / fy;
        			double r2 = x*x + y*y;

        			double dx = 2 * p1*x*y + p2*(r2 + 2 * x*x);
        			double dy = p1*(r2 + 2 * y*y) + 2 * p2*x*y;
        			double scale = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2);

        			double xBis = x*scale + dx;
        			double yBis = y*scale + dy;

        			double xCorr = xBis*fx + cx;
        			double yCorr = yBis*fy + cy;

        			if (xCorr >= 0 && xCorr < uPhoto.cols && yCorr >= 0 && yCorr < uPhoto.rows) //aplicacion de correccion y calibracion
        				{
            				uPhoto.at<uchar>(yCorr, xCorr) = frame.at<uchar>(j, i); /*ajustar la imagen corregida uPhoto en base a dimenciones de la original frame, por eso es que se debe clonar.*/
        				}
    				}
			}
		
		
  		Mat thres; //Matriz para almacenar la imagen despues de aplicar el OTSU y la BINARIZACION INVERTIDA
  		threshold(uPhoto, thres, 0, 255,THRESH_OTSU + THRESH_BINARY_INV); //CV_THRESH_BINARY | CV_THRESH_OTSU //OTSU Y BINARIZACION INVERTIDA
 
 //***************************AQUI EMPIEZA EL ETIQUETADO DE COMPONENTES CONECTADOS*******************************************************************
 //connectedComponents guarda las etiquetas en la matriz labelImage
  
    		Mat labelImage(thres.size(), CV_32S);
    //conectedComponents regresa el numero de etiquetas encontradas y se almacenan en nLabels
    		int nLabels = connectedComponents(thres, labelImage, 8, CV_32S);
    		std::vector<Vec3b> colors(nLabels);
    		colors[0] = Vec3b(0, 0, 0);//Definicion del valor cero para el vector de los colores de los componentes que sera el del fondo de la imagen
     
     		cout<<nLabels<<"/n";
   //En este for se asignan los colores a cada objeto etiquetado de acuerdo a las label que genera la funcion en linea
    		for(int label = 1; label < nLabels; label++)
			{
      			colors[label] = Vec3b( (120*label), (50*label), (200*label ));
    			}
	
    //Enseguida cambian de color cada objeto detectado y el resultado se cuarda en la imagen "Coloreada" con su color arriba asignado
    		Mat Coloreada(thres.size(), CV_8UC3);
    		for(int r = 0; r < Coloreada.rows; ++r)
			{ //Algoritmo para asignar los colores a cada pixel que conforma el objeto etiquetado
        		for(int c = 0; c < Coloreada.cols; ++c)
        			{
            			int label = labelImage.at<int>(r, c);
            			Vec3b &pixel = Coloreada.at<Vec3b>(r, c);
            			pixel = colors[label];
        			//cout<<label<<endl; //Habilitarlo si deseo que me muestre el indice de etiqueta que esta coloreando en linea
         			}
     			}	

		//cout<<nLabels<<endl; //Habilitarlo si deseo que me muestre en la terminal la cantidad de objetos que etiqueta la funcion

		vector<vector<Point> > contours; // Vectores para el Algoritmo de extraccion de bordes
  		vector<Vec4i> hierarchy;
  		findContours( thres.clone(), contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );//Extraccion de bordes

  		Mat drawing = Mat::zeros( frame.size(), CV_8UC3 ); // En drawing se guardara el dibujo final que es el que se muestra con el rectangulo
  		Scalar color = Scalar( 255, 255, 255 );
  		vector<vector<Point> > contours_poly( contours.size() );

  		vector<Rect> boundRect( contours.size() );

  		vector<RotatedRect> minEllipse( contours.size() );
   
  		for( int i = 0; i< contours.size(); i++ )
			{  //Calculo de caracteristicas
      			p = arcLength(contours[i],true); 
      			a = contourArea( contours[i]);
  
			 
			if(a>200) //CAMBIOS AGREGADOS SOLO CONSIDERA OBJETOS MAYORES A 1000 PIXELES
				{
      				drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );

      				de = sqrt(4*a/3.1416);

      				boundRect[i] = boundingRect(contours[i]); // Dibuja el rectangulo de acuerdo a los bordes extraidos
      				rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0 );
      				double rect_area = boundRect[i].width*boundRect[i].height;
      				extent = a/rect_area;
			
					//Guardar caracteristicas geometricas en matriz: caracteristicas que por cuestion de tiempo es de tipo bidimencional array
				
							    // La cantidad de columnas por las caracteristicas empleadas al momento
      				if( contours[i].size()> 5) // Discriminar los contornos de los objetos que dibujara
					{
        				minEllipse[i] = fitEllipse(contours[i]);
        				//ellipse( drawing, minEllipse[i], color, 1, 8 );
      				}
      				if(a!=0)					
					{
				
					//for(label2=1;label2<nLabels;label2++) //Extraccion de caracteristicas de cada objeto etiquetado en linea
					//{
					for(int carac=0;carac<6;carac++)
						{
						if(carac==0)
						caracteristicas[i][carac]=a;//AREA
						if(carac==1)
						caracteristicas[i][carac]=p;//PERIMETRO
						if(carac==2)
						caracteristicas[i][carac]=minEllipse[i].size.height;//SEMIEJE MAYOR
						if(carac==3)
						caracteristicas[i][carac]=minEllipse[i].size.width;//SEMIEJE MENOR
						if(carac==4)
						caracteristicas[i][carac]=de;//DIAMETRO EQUIVALENTE
						if(carac==5)
						caracteristicas[i][carac]=extent;//EXCENTRICIDAD
												
						}
					
					}
					
					getOrientation(contours[i], src, a, p, de, extent);//FUNCION PARA EL PCA Y DIBUJAR LAS ETIQUETAS

				cuenta_datos++;
      				}
			//CAMBIOS AGREGADOS UN IF A>1000 PARA DISCRIMINAR OBJETOS MENORES A 1000 PIXELES
   			}
  			
		imshow("algoritmo",drawing);//MOSTRAR LAS IMAGENES EN LINEA DE LA APLICACION DEL ALGORITMO CON CADA OBJETO EN UN RECTANGULO
		imshow("binarizada",thres);//MOSTRAR LA IMAGEN CON SEGMENTACION BI CLASE
		imshow("Componentes Conectados",Coloreada);//MOSTRAR IMAGENES EN LINEA DE LA SEGMENTACION MULTICLASE CON COMPONENTES CONECTADOS
		
		if(waitKey(30) >= 0) break;//PRESIONAR ALGUNA TECLA PARA SALIR DE LA EJECUCION DEL FOR INFINITO		

		}		
	
	return 0;
	}
