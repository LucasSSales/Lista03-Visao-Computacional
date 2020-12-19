import numpy as np
import cv2

#Função apenas para gerar uma string do ultimo item do vector
def stringando(vec):
    s = "[ "
    for v in vec[-1]:
        s += str(v) + " "
    s += "]"
    return s

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Objeto para capturar os frames da webcam
cap = cv2.VideoCapture(0)
# Objetos para salvar o video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 4.0, (640, 480))

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while(True):
    # Capturando os frames
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectando o tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    rs, ts = "", ""
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, (7,6), corners2,ret)
        # Obtendo os vetores de calibração
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        rs, ts = stringando(rvecs), stringando(tvecs)
    
    # Escrita na tela dos dados
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "ROTATION: " + rs, (10,450), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "TRANSLATION: " + ts, (10,470), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Exibindo na tela
    cv2.imshow('Lista 03 - Questão 03',frame)
    # Salvando o video com os vetores em tela
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()