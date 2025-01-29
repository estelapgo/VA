import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
 
# Estela Pillo González

# Alteración del rango dinámico
def adjustIntensity(inImage, inRange=[], outRange=[0,1]):
    width, height = inImage.shape   # Cálculo de las dimensiones

    outImage = np.zeros((width, height))

    if not inRange:
        min = np.min(inImage)
        max = np.max(inImage) 
    else:
        min = inRange[0]
        max = inRange[1]

    for i in range(width):
        for j in range(height):
            outImage[i,j] = outRange[0] + (((outRange[1] - outRange[0]) * (inImage[i,j] - min))/(max - min))
    
    return outImage 

# Ecualización de histograma
def equalizeIntensity(inImage, nBins=256):
    m, n = inImage.shape  

    # Cálculo del histograma
    hist = cv.calcHist([(inImage * 255).astype(np.uint8)], [0], None, [nBins], [0, 256])

    # Histograma acumulado Hc(g)
    hist_acum = np.cumsum(hist)

    # Transformación T(g) 
    T = (hist_acum/(m*n)*255)

    # Transformación de la imagen original con T
    outImage = T[(inImage * 255).astype(np.uint8)]

    return outImage

# Filtro espacial mediante convolución
def filterImage(inImage, kernel):
    m, n = inImage.shape # Dimensiones de la imagen
    P, Q = kernel.shape  # Dimensiones con el kernel del filtro de entrada

   
    # Aplico padding a la imagen
    # Uso edge para que, al extender el borde hacia fuera, use el valor del píxel más cercano al borde
    padded_inImage = np.pad(inImage, ((P //2, P // 2), (Q // 2, Q // 2)), mode = 'edge')
    
    outImage = np.zeros((m, n))

    # Convolución
    for i in range(m):
        for j in range(n):
            outImage[i, j] = np.sum(padded_inImage[i:i + P, j:j + Q] * kernel)
    
    return outImage

    
# Kernel Gaussiano unidimensional
def gaussKernel1D(sigma): 
    N = 2 * int(np.ceil(3 * sigma)) + 1

    # Centro de la gaussiana
    center = N // 2     

    # Creo el kernel, inicializándolo a ceros
    kernel = np.zeros(N)

    for i in range(N):
        kernel[i] = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-((i - center) ** 2) / (2 * sigma ** 2 ))

    return kernel 

# Suavizado Gaussiano bidimensional
def gaussianFilter(inImage, sigma):

    kernel = gaussKernel1D(sigma) 
    kernel2D = kernel[np.newaxis, :]                          # Creación del kernel 2D a partir del kernel 1D
    tmpImage = filterImage(inImage, kernel2D)                 # Convolución horizontal, con el kernel 2D
    outImage = filterImage(tmpImage, kernel2D.reshape(-1, 1)) # Convolución vertical, con el kernel traspuesto

    return outImage

# Filtro de medianas bidimensional
def medianFilter(inImage, filterSize):
    m, n = inImage.shape 

    padding = filterSize // 2

    inImage_padded = np.pad(inImage, padding, mode='edge')

    outImage = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            window = inImage_padded[i:i + filterSize, j:j + filterSize]
            
            # Cálculo de la mediana de la ventana
            outImage[i, j] = np.median(window)
    
    return outImage

# Operadores morfológicos: erosión, dilatación, apertura y cierre

# Erosión
def erode(inImage, SE, center=[]):
    m, n = inImage.shape  
    SE_h, SE_w = SE.shape  # Dimensiones del elemento estructurante

    if not center:
        center = (SE_h // 2, SE_w // 2)
    
    outImage = np.zeros((m, n))
    
    # Cálculo del padding necesario para la imagen
    pad_h = center[0]
    pad_w = center[1]

    # Aplicación del padding
    padded_inImage = np.pad(
        inImage, 
        ((pad_h, SE_h - pad_h - 1), (pad_w, SE_w - pad_w - 1)), 
        mode='constant', 
        constant_values=0
    )

    for i in range(m):
        for j in range(n):
            # Ventana del tamaño del SE
            window = padded_inImage[i:i + SE_h, j:j + SE_w]

            # Aplicar la erosión: si todos los píxeles bajo el SE son 1, poner 1 en la salida
            if np.all(window[SE == 1] == 1):
                outImage[i, j] = 1
            else:
                outImage[i, j] = 0

    return outImage

# Dilatación
def dilate(inImage, SE, center = []):
    # Si no se proporciona un centro, lo defino
    if not center:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]
    else:
        center = [center[0], center[1]]
    
    # La imagen de salida es creada con una copia de la original
    outImage = np.copy(inImage)
    
    # Iteración sobre cada píxel de la imagen
    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):
            if inImage[i, j] == 1:
                for x in range(SE.shape[0]):        # Itero sobre el SE
                    for y in range(SE.shape[1]):
                        if SE[x, y] == 1:
                            nx, ny = i + x - center[0], j + y - center[1]
                            if 0 <= nx < inImage.shape[0] and 0 <= ny < inImage.shape[1]:
                                outImage[nx, ny] = 1
    return outImage

# Apertura
def opening(inImage, SE, center = []): # Primero se aplica erosión y luego dilatación

    erodedImage = erode(inImage, SE, center)
    openedIMage = dilate(erodedImage, SE, center)

    return openedIMage

# Cierre
def closing(inImage, SE, center = []): # Primero se aplica dilatación y luego erosión

    dilatedImage = dilate(inImage, SE, center)
    closedImage = erode(dilatedImage, SE, center)

    return closedImage

# Llenado morfológico de regiones
def fill(inImage, seeds):
 
    # La imagen de salida es creada con una copia de la original
    outImage = inImage.copy()
    
    # Inicialización de los puntos a procesar
    queue = seeds.tolist()
    
    while queue:
        x, y = queue.pop(0)
        
        # Se mira si el punto está en la imagen y si es un píxel blanco
        if 0 <= x < outImage.shape[0] and 0 <= y < outImage.shape[1] and outImage[x, y] == 0:
            outImage[x, y] = 255
            
            # Se agregan los vecinos a la lista
            queue.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
    
    return outImage

# Gradiente de una imagen
def gradientImage(inImage, operator):

    if operator == 'Roberts':
        op_gx = np.array([
            [-1,0],
            [0,1]
        ])

        op_gy = np.array([
            [0,-1],
            [1,0]
        ])
    elif operator == 'CentralDiff':
        op_gx = np.array([
           [-1,0,1] 
        ])

        op_gy = np.array([
            [-1],
            [0],
            [1]
        ])
    elif operator == 'Prewitt':
        op_gx = np.array([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
        ])

        op_gy = np.array([
            [-1,-1,-1],
            [0,0,0],
            [1,1,1]
        ]) 
    elif operator == 'Sobel':
        op_gx = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ]) 

        op_gy = np.array([ 
            [-1,-2,-1],
            [0,0,0],
            [1,2,1]
        ]) 

    else:
        raise ValueError("Operador inválido. Pruebe a usar Roberts, CentralDiff, Prewitt o Sobel")

    gx = filterImage(inImage,op_gx)
    gy = filterImage(inImage,op_gy)

    return gx,gy

# Filtro Laplaciano de Gaussiano
def LoG(inImage, sigma):

    kernel = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])
    
    smoothed_image = gaussianFilter(inImage,sigma)

    outImage = filterImage(smoothed_image,kernel)

    return outImage

# Función auxiliar: cálculo de la supresión no máxima en Canny
def notMaxSupp(magnitud,direc):
    mag_supp = np.zeros_like(magnitud)

    for i in range(1,magnitud.shape[0]-1):
        for j in range(1,magnitud.shape[1]-1):
            direction = direc[i,j]
            m1,m2 = magnitud[i+1,j],magnitud[i-1,j]

            if((direction <= np.pi/4 and direction >= -np.pi/4) or (direction >= 3*np.pi/4) or (direction <= -3*np.pi/4)):
                m1,m2 = magnitud[i,j+1],magnitud[i,j-1]

            if magnitud[i,j] >= m1 and magnitud[i,j] >= m2:
                mag_supp[i,j] = magnitud[i,j]
    
    return mag_supp

# Detector de bordes de Canny
def edgeCanny(inImage,sigma,tlow,thigh):
    # Aplicación del filtro gaussiano
    tmpImage = gaussianFilter(inImage,sigma)

    # Se obtiene gx, gy, magnitud y dirección
    gx,gy = gradientImage(inImage,'Sobel')
    magnitud = np.sqrt(gx**2 + gy**2)
    direc = np.arctan2(gy,gx)

    # Uso de la función auxiliar
    mag_supp = notMaxSupp(magnitud,direc)

    strong = mag_supp > thigh 
    weak = (mag_supp >= tlow) & (mag_supp <= thigh)

    outImage = np.zeros_like(mag_supp)

    outImage[strong] = 1

    for i in range(1, outImage.shape[0] - 1):
        for j in range(1 , outImage.shape[1] - 1):
            if weak[i,j]:
                if(outImage[i+1,j-1:j+1].max() or
                   outImage[i,j-1:j+1].max() or
                   outImage[i-1,j-1:j+1].max()):

                   outImage[i,j] = 1

    return outImage 

#TESTS

# Función auxiliar para mostrar imágenes e histogramas
def mostrar_imagenes_y_histogramas(inImage, outImage, titulo_original, titulo_resultado):
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Resultados")

    def cerrar_ventana(event):
        if event.key == '0':  
            plt.close(fig)  

    fig.canvas.mpl_connect('key_press_event', cerrar_ventana)

    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(inImage, cv.COLOR_BGR2RGB))  
    plt.title(f"{titulo_original}")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    if len(outImage.shape) == 2:  
        plt.imshow(outImage, cmap='gray')
    else:  
        plt.imshow(cv.cvtColor(np.uint8(outImage * 255), cv.COLOR_BGR2RGB))
    plt.title(f"{titulo_resultado}")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.hist(inImage.ravel(), bins=256, color='gray', alpha=0.7)
    plt.title(f"Histograma de {titulo_original}")
    plt.xlim([0, 256])

    plt.subplot(2, 2, 4)
    plt.hist(np.uint8(outImage * 255).ravel(), bins=256, color='gray', alpha=0.7)
    plt.title(f"Histograma de {titulo_resultado}")
    plt.xlim([0, 256])

    plt.tight_layout()

    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()

# Test para ajustar la intensidad
def testAdjustIntensity():
    inImage = cv.imread('grays.png', cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    inImageNorm = inImage / 255.0

    outImage = adjustIntensity(inImageNorm, [], [0.1, 0.9])

    mostrar_imagenes_y_histogramas(inImage, outImage, 'grays.png', 'imagen_rdinamico.png')

    cv.imwrite('salidas/imagen_rdinamico.png', np.uint8(outImage * 255))

# Test para ecualización de intensidad
def testEqualizeIntensity():
    inImage = cv.imread('grays.png', cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    inImageNorm = inImage / 255.0

    outImage = equalizeIntensity(inImageNorm)

    mostrar_imagenes_y_histogramas(inImage, outImage, 'grays.png', 'imagen_ecualizada.png')

    cv.imwrite('salidas/imagen_ecualizada.png', outImage)

#Test para probar el filtrado especial mediante convolución
def testFilterImage(): 
    inImage = cv.imread('circles.png', cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    inImageNorm = inImage / 255.0

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # kernel = np.array([
    #     [1,1,1],
    #     [1,1,1],
    #     [1,1,1]
    # ])/9.0

    outImage = filterImage(inImageNorm, kernel)

    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen filtrada')

    cv.imwrite('salidas/imagen_filtrada.png', np.uint8(outImage * 255))

# Test para probar el kernel Gaussiano unidimensional
def testGaussKernel1D():
    sigma = 1.0         
    gaussian_kernel = gaussKernel1D(sigma)
    print(gaussian_kernel)
    # Los valores más cercanos al centro tienen mayor intensidad

# Test para probar el suavizado Gaussiano bidimensional
def testGaussianFilter():
    inImage = cv.imread('chica.jpeg', cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    inImageNorm = inImage / 255.0
    sigma = 2
    outImage = gaussianFilter(inImageNorm, sigma)

    cv.imwrite('salidas/imagen_gauss.png', np.uint8(outImage * 255))

    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen Gauss')

# Test para probar el filtro de medianas bidimensional
def testMedianFilter():
    inImage = cv.imread('chica.jpeg',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    inImageNorm = inImage / 255.0
    filterSize = 7

    outImage = medianFilter(inImageNorm,filterSize)

    cv.imwrite('salidas/imagen_medianas.png',np.uint8(outImage * 255))  
    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen medianas')

# Test para probar la erosión
def testErode():
    inImage = cv.imread('morph.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    kernel = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])

    center = [0,0]

    inImageNorm = inImage / 255

    outImage = erode(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_erosion.png',outImage * 255)  
    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen erosionada')

# Test para probar la dilatación
def testDilate():
    inImage = cv.imread('morph.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    center = [0,0]

    inImageNorm = inImage / 255
    outImage = dilate(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_dilatacion.png',outImage * 255) 
    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen dilatada')


# Test para probar la apertura 
def testOpening():
    inImage = cv.imread('dilatacion.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    center = [0,0]

    inImageNorm = inImage / 255

    outImage = opening(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_apertura.png',outImage * 255)
    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen apertura')

# Test para probar el cierre
def testClosing():
    inImage = cv.imread('cierre.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    center = [0,0]

    inImageNorm = inImage / 255

    outImage = closing(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_cierre.png',outImage * 255) 
    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen cierre')

# Función auxiliar para calcular la magnitud
def calcularMagnitud(gx,gy):
    height,width = gx.shape[:2]

    outImage = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            outImage[i,j] = math.sqrt(pow(gx[i,j],2) + pow(gy[i,j],2))

    return outImage

# Test para probar el llenado morfológico de regiones
def testFill():
    inImage = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255
    
    seeds = np.array([[3, 3]])
    
    outImage = fill(inImage, seeds)

    cv.imwrite('salidas/imagen_fill.png',outImage) 
    
    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen Fill')


# Función para probar el gradiente de una imagen
def testGradientImage():
    inImage = cv.imread('circles.png', cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    inImageNorm = inImage / 255

    operator = 'Sobel'

    gx,gy = gradientImage(inImageNorm,operator)

    outImage = calcularMagnitud(gx,gy)

    cv.imwrite('salidas/imagen_gx_' + operator + '.png',gx * 255) 
    cv.imwrite('salidas/imagen_gy_' + operator + '.png',gy * 255)
    cv.imwrite('salidas/imagen_magnitud_' + operator + '.png',outImage * 255)

    cv.imshow('Imagen gx - ' + operator, gx)
    cv.imshow('Imagen gy - ' + operator, gy)
    cv.imshow('Imagen resultado - ' + operator, outImage)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

# Test para probar el filtro Laplaciano de Gaussiano
def testLoG():
    inImage = cv.imread('chica.jpeg', cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    min = np.min(inImage)
    max = np.max(inImage)

    inImageNorm = 2 * ((inImage - min) / (max - min)) -1 

    outImage = LoG(inImageNorm,0.8)

    cv.imwrite('salidas/imagen_LoG.png',outImage * 255) 

    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen Log')

# Test para probar el detector de bordes de Canny
def testCanny():
    inImage = cv.imread('chica.jpeg',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imagen"

    inImageNorm = inImage / 255

    outImage = edgeCanny(inImageNorm,1.5,0.1,0.5)

    cv.imwrite('salidas/imagen_Canny.png',outImage * 255) 
    mostrar_imagenes_y_histogramas(inImage, outImage, 'Imagen original', 'Imagen Canny')


def main():

    testAdjustIntensity() 
    testEqualizeIntensity() 
    testFilterImage() 
    testGaussKernel1D()
    testGaussianFilter() 
    testMedianFilter()
    testErode()
    testDilate()
    testOpening()
    testClosing()
    testFill()
    testGradientImage()
    testLoG()
    testCanny()

if __name__ == "__main__":
    main()