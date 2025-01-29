import cv2
import os
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import active_contour

output_folders = [
    'salidas',
    'salidas/pestanas',
    'salidas/brillospupila',
    'salidas/irispupila',
    'salidas/esclerotica'
]

for folder in output_folders:
    os.makedirs(folder, exist_ok=True)
    
def detectar_pupila_iris(image):
    # Detecto círculos en la imagen que correspondan a pupila e iris
    circles_pupil = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=40)
    circles_iris = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=30, maxRadius=60)
  
    if circles_pupil is not None:
        circles_pupil = np.round(circles_pupil[0, :]).astype("int")
        pupil = circles_pupil[0]

    if circles_iris is not None:
        circles_iris = np.round(circles_iris[0, :]).astype("int")
        iris = circles_iris[0]

    return pupil, iris

def dibujar_parabola(image, top_left, bottom_right, orientation="U"):
    # Dibujo una parábola según puntos y orientación que se indica
    points = obtener_puntos_parabola(top_left, bottom_right, orientation)
    for x, y in points:
        if top_left[1] <= y < bottom_right[1]:
            image[y, x] = (255, 255, 0)

def obtener_puntos_parabola(top_left, bottom_right, orientation="U"):
    # Calculo los puntos de la parábola
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    points = []

    a = 4 * height / (width * width)
    for x in range(top_left[0], bottom_right[0]):
        normalized_x = x - top_left[0]
        y_offset = int(a * (width / 2 - normalized_x) ** 2)
        y = bottom_right[1] - y_offset if orientation == "U" else top_left[1] + y_offset
        points.append((x, y))
    return points

def detectar_brillos_pupila(inImage, pupil):
    cv2.GaussianBlur(inImage, (5, 5), 0)
    
    # Ajusto el contraste de la imagen y detecto brillos dentro de la pupila
    contraste = np.array(255/np.log(1 + np.max(inImage)) * np.log(1 + inImage))
    
    contraste_pupila = np.logical_not(np.where(contraste > 160, 1, 0)).astype(np.uint8)

    # Creo una máscara para delimitar la pupila
    mask = np.zeros_like(contraste_pupila)
    if pupil is not None:
        cv2.circle(mask, (pupil[0], pupil[1]), pupil[2], 1, thickness=-1)

    contraste_pupila = cv2.bitwise_and(contraste_pupila, mask)
    
    contornos, _ = cv2.findContours(contraste_pupila, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    color_image = cv2.cvtColor(inImage, cv2.COLOR_GRAY2BGR)

    # Dibujo los contornos pequeños en rojo
    for c in contornos:
        if cv2.contourArea(c) < 500:
            cv2.drawContours(color_image, [c], 0, (0, 0, 255), 1)  # Contorno de brillos en rojo con grosor 2
    
    return color_image
   
def detectarPestanas(inImage, pupil, iris):

    #Dibujo los contornos de pupila e iris en la imagen, para facilitar el trabajo a Canny
    if pupil is not None:
        cv2.circle(inImage, (pupil[0], pupil[1]), pupil[2], 255, 1) 
    if iris is not None:
        cv2.circle(inImage, (iris[0], iris[1]), iris[2], 255, 1)  

    # Creo una máscara para no detectar pestañas en zonas muy superiores en la imagen
    mask = np.ones_like(inImage, dtype=np.uint8) * 255
    if iris is not None:
        limite_superior = max(0, iris[1] - iris[2] - 25)  
        mask[0:limite_superior, :] = 0

    # Aplico Canny para detectar las pestañas
    edges= cv2.Canny(inImage, 100, 200)
    canny = cv2.bitwise_and(edges, edges, mask=mask)

    return canny


def detectar_esclerotica(inImage, top_parabola_points, bottom_parabola_points, iris):
    
    # Detecto la región de la esclerótica delimitada por dos parábolas y el iris
    mask = np.zeros(inImage.shape[:2], dtype=np.uint8)

    # Creo un contorno a partir de las parábolas
    top_points = np.array(top_parabola_points, dtype=np.int32)
    bottom_points = np.array(bottom_parabola_points, dtype=np.int32)
    contour_points = np.vstack((top_points, bottom_points[::-1]))
    cv2.fillPoly(mask, [contour_points], 255)

    if iris is not None:
        iris_mask = np.zeros_like(mask)
        cv2.circle(iris_mask, (iris[0], iris[1]), iris[2], 255, -1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(iris_mask))

    thresh = cv2.adaptiveThreshold(inImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    kernel = np.ones((5, 5), np.uint8)
    sclera_region = cv2.morphologyEx(cv2.bitwise_and(mask, thresh), cv2.MORPH_CLOSE, kernel)

    return sclera_region

def procesar_imagen(filename, input_folder, output_folder):

    image = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (11, 11), 2)

    pupil, iris = detectar_pupila_iris(blurred)
    brillos = detectar_brillos_pupila(image, pupil)
    pestanas = detectarPestanas(image, pupil, iris)

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if pupil is not None:
        cv2.circle(color_image, (pupil[0], pupil[1]), pupil[2], (255, 0, 0), 2)
    if iris is not None:
        cv2.circle(color_image, (iris[0], iris[1]), iris[2], (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_folder, 'irispupila', f"iris_pupila_{filename}"), color_image)
    cv2.imwrite(os.path.join(output_folder, 'brillospupila', f"brillos_{filename}"), brillos)
    cv2.imwrite(os.path.join(output_folder, 'pestanas', f"pestanas_{filename}"), pestanas)

    if pupil is not None and iris is not None:
        roi_left = int(iris[0] - iris[2] * 2.2)
        roi_right = int(iris[0] + iris[2] * 2.2)
        roi_top = int(iris[1] - iris[2])
        roi_bottom = int(iris[1] + iris[2] * 1.3)
        mid_y = (roi_top + roi_bottom) // 2

        top_parabola = obtener_puntos_parabola((roi_left, roi_top), (roi_right, mid_y), orientation="inverted_U")
        bottom_parabola = obtener_puntos_parabola((roi_left, mid_y), (roi_right, roi_bottom), orientation="U")
        sclera_region = detectar_esclerotica(image, top_parabola, bottom_parabola, iris)

        sclera_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        sclera_image[sclera_region > 0] = [0, 255, 255]
        cv2.circle(sclera_image, (iris[0], iris[1]), iris[2], (0, 255, 0), 2)
        cv2.circle(sclera_image, (pupil[0], pupil[1]), pupil[2], (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_folder, 'esclerotica', f"esclerotica_{filename}"), sclera_image)

        roi = image[roi_top:roi_bottom, roi_left:roi_right]
        image[roi_top:roi_bottom, roi_left:roi_right] = cv2.equalizeHist(roi)

        cv2.rectangle(color_image, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)
        cv2.line(color_image, (roi_left, mid_y), (roi_right, mid_y), (0, 0, 255), 2)
        dibujar_parabola(color_image, (roi_left, roi_top), (roi_right, mid_y), orientation="inverted_U")
        dibujar_parabola(color_image, (roi_left, mid_y), (roi_right, roi_bottom), orientation="U")
        cv2.imwrite(os.path.join(output_folder, f"resultado_{filename}"), color_image)
    else:
        print(f"No se encontró pupila ni iris en {filename}")

input_folder = 'entradas'
output_folder = 'salidas'   

for filename in os.listdir(input_folder):
    if filename.endswith('.bmp'):
       procesar_imagen(filename, input_folder, output_folder)
   