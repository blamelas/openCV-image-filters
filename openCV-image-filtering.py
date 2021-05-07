import cv2
import numpy as np
from matplotlib import pyplot as plt

####### Mini Projeto Bruno Lamelas ############
###### IMPORTANTE IMPORTAR O "opencv-python-headless a partir das prefêrencias" ##############


########################################
# Ler Imagem original
img_path = "resources/quadrado5.jpg"
img = cv2.imread(img_path)
print("Imagem Lida")

# Mostrar imagem original
cv2.imshow('Imagem Original', img)
print("Imagem Original Mostrada")
cv2.waitKey()
cv2.destroyAllWindows()
########################################
# Converter para Gray Scale e mostrar
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem GrayScale', gray)
print("Imagem GrayScale Mostrada")
cv2.waitKey()
cv2.destroyAllWindows()

########################################
# Remover Ruido Aplicando Blur
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
cv2.imshow('Imagem GrayScale Com Blur', img_gaussian)
print("Imagem GrayScale Com Blur Mostrada")
cv2.waitKey()
cv2.destroyAllWindows()

#######################################
## Canny
img_canny = cv2.Canny(img,100,200)
cv2.imshow("Imagem Resultado Canny", img_canny)
cv2.waitKey()

#######################################
## Sobel
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely
cv2.imshow("Imagem Resultado Sobel X", img_sobelx)
cv2.imshow("Imagem Resultado Sobel Y", img_sobely)
cv2.imshow("Imagem Resultado Sobel", img_sobel)
cv2.waitKey()

#######################################
## prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
cv2.imshow("Imagem Resultado Prewitt X", img_prewittx)
cv2.imshow("Imagem Resultado Prewitt Y", img_prewitty)
cv2.imshow("Imagem Resultado Prewitt", img_prewittx + img_prewitty)
cv2.waitKey()


#######################################
# Lapacian
imgLaplacian = cv2.Laplacian(img_gaussian,cv2.CV_64F)
sobelx = cv2.Sobel(imgLaplacian,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(imgLaplacian,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img_gaussian,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(imgLaplacian,cmap = 'gray')
plt.title('Img Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
print("Imagem Resultado Lapacian")
cv2.waitKey()


#######################################

cv2.waitKey(0)
cv2.destroyAllWindows()

