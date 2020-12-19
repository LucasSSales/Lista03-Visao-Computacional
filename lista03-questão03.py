import numpy as np
import cv2

def show_vectors(vec, string):
    print("VETOR DE "+string)
    for v in vec:
        print("[", end="")
        for i in v:
            print(i, end="")
            print(" ", end="")
        print("]")
    print("")

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# pega os frame
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 4.0, (640, 480))

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


#640.0 w
#480.0 h

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #if ret:
    #    out.write(frame)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    ret, corners = cv2.findChessboardCorners(frame, (7,6),None)

    #print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print(cap.get(cv2.CAP_PROP_FPS))

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        gray = cv2.drawChessboardCorners(gray, (7,6), corners2,ret)
        frame = cv2.drawChessboardCorners(frame, (7,6), corners2,ret)


        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        show_vectors(rvecs, "ROTAÇÃO")
        show_vectors(tvecs, "TRANSLAÇÃO")
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()