#copy following Python code to initiate the EYEternal Desktop Application

GUI
import blink
import distance
import gameObject
import gameCursor
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()

window.geometry("900x640")
window.title('EYEternal')
window.configure(bg = "#7CCEB5")

canvas = Canvas(
    window,
    bg = "#7CCEB5",
    height = 640,
    width = 900,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    390.0,
    15.0,
    882.0,
    625.0,
    fill="#4EBC59",
    outline="")

canvas.create_text(
    39.0,
    75.0,
    anchor="nw",
    text="EYEternal",
    fill="#49549E",
    font=("RobotoRoman Bold", 70 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    195.0,
    376.0,
    image=image_image_1
)

canvas.create_rectangle(
    429.0,
    31.0,
    843.0,
    118.0,
    fill="#366B3E",
    outline="")

canvas.create_text(
    455.0,
    52.0,
    anchor="nw",
    text=" EYEternal Programs",
    fill="#FFFFFF",
    font=("RobotoRoman Bold", 40 * -1)
)

#blink
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: blink.blinkCounter(),
    relief="flat"
)

button_1.place(
    x=412.0,
    y=141.0,
    width=447.0,
    height=67.0
)

#distance
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: distance.distanceCounter(),
    relief="flat"
)
button_2.place(
    x=412.0,
    y=250.0,
    width=447.0,
    height=67.0
)

#dynamic objet
button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: gameObject.object(),
    relief="flat"
)
button_3.place(
    x=412.0,
    y=359.0,
    width=447.0,
    height=67.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: gameCursor.cursor(),
    relief="flat"
)
button_4.place(
    x=412.0,
    y=468.0,
    width=447.0,
    height=67.0
)

canvas.create_text(
    540.0,
    574.0,
    anchor="nw",
    text="(Press ‘q’ to quit)",
    fill="#FFFFFF",
    font=("RobotoRoman Bold", 25 * -1)
)

window.resizable(False, False)
window.mainloop()
blink (Blinking Rate Counter)
import cv2
import time
import cvzone
import numpy as np
from scipy.spatial import distance as dist
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

def blinkCounter():
    #read web cam
    cam = cv2.VideoCapture(0)
    #restrict to only one face
    detector = FaceMeshDetector(maxFaces=1)
    plotY = LivePlot(640, 360, [0.1, 0.4], invert=True)

    #landmark id for eyes
    idList = [130, 160, 158, 243, 22, 24, 398, 385, 387, 446, 390, 380]

    earList = []
    blinkCounter = 0
    counter = 0
    color = (255, 0, 255)

    start = time.time()

    def error_checking(eye):
        # 1) user not looking at teh screen at all
        if eye[0] is None:
            if eye[1] is None:
                if eye[2] is None:
                    if eye[3] is None:
                        if eye[4] is None:
                            if eye[5] is None:
                                return eye

        #2) only a part of user's eye is captured by computer
        #2-1) upper part of eye is not captured
        if eye[1] is None and eye[2] is None:
            xEye1 = eye[5][0]
            yEye1 = 2*eye[0][1] - eye[5][1]
            eye[1] = (xEye1, yEye1)

            xEye2 = eye[4][0]
            yEye2 = 2*eye[3][1] - eye[4][1]
            eye[2] = (xEye2, yEye2)

        # 2-2) lower part of eye is not captured
        if eye[4] is None and eye[5] is None:
            xEye4 = eye[2][0]
            yEye4 = 2*eye[3][1] - eye[2][1]
            eye[4] = (xEye4, yEye4)

            xEye5 = eye[1][0]
            yEye5 = 2*eye[0][1] - eye[1][1]

        return eye

    def specific_eye_aspect_ratio(eye):
        d1 = dist.euclidean(eye[1], eye[5])
        d2 = dist.euclidean(eye[2], eye[4])
        d3 = dist.euclidean(eye[0], eye[3])
        # eye aspect ratio = (||p1 - p5|| + ||p2 - p4||)/2||p0-p3||
        ear = (d1 + d2) / (2.0 * d3)
        return ear

    while True:
        success, img = cam.read()
        #getting face landmarks
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            #1. face distance
            faceD = faces[0]
            pointLeft = faceD[145]
            pointRight = faceD[374]

            l, _ = detector.findDistance(pointLeft, pointRight)
            L = 6.3
            # 2.1) find the focal length, f
            # d = 50
            # f = (l*d)/L

            # 2.2) find the face distance, d
            f = 840
            d = (L * f) / l

            #2. blink detector
            face = faces[0]
            #marking landmark id for eyes
            for id in idList:
                cv2.circle(img, face[id], 5, color, cv2.FILLED)

            #finding distance between eye lids
            leftE = [face[130], face[160], face[158], face[243], face[22], face[24]]
            rightE = [face[398], face[385], face[387], face[446], face[390], face[380]]

            #leftE = error_checking(leftE)
            #rightE = error_checking(rightE)

            leftEAR = specific_eye_aspect_ratio(leftE)
            rightEAR = specific_eye_aspect_ratio(rightE)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            earList.append(ear)
            if len(earList) > 5:
                earList.pop(0)
            earAvg = float(sum(earList) / len(earList))
            #print(earAvg)

            EYE_AR_THRESH = 0.25
            EYE_AR_THRESH = EYE_AR_THRESH + ((40-d)*0.0015)
            #print("T", EYE_AR_THRESH)

            if earAvg < EYE_AR_THRESH and counter == 0:
                blinkCounter += 1
                color = (0, 200, 0)
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (255, 0, 255)

            end = time.time()

            dur = int(end - start) + 1

            blinkRate = int(60*blinkCounter/dur)

            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 50), colorR=color)

            cvzone.putTextRect(img, f'Timer: {dur} sec', (50, 100), colorR=color)

            cvzone.putTextRect(img, f'Blinking Rate: {blinkRate}/min', (50, 150), colorR=color)

            """
            cvzone.putTextRect(img, f'Distance: {int(d)}cm',
                               (face[10][0] - 150, face[10][1] - 20),
                               scale=2.5, colorR=color)
            """

            alert = np.zeros((512, 512, 3), dtype="uint8")
            alert = cv2.resize(alert, (1280, 180))

            if blinkRate < 20:
                blinkMore = int(1+(1/3)*dur-blinkCounter)
                cvzone.putTextRect(alert, f'Blink {blinkMore} Times More', (450, 90), colorR=(0, 0, 255))
            else:
                cvzone.putTextRect(alert, 'Safe Blinking Rate', (450, 90), colorR=(0, 255, 0))
            imgPlot = plotY.update(earAvg, color)
            img = cv2.resize(img, (640, 360))
            imgStock = cvzone.stackImages([img, imgPlot], 2, 1)
            imgStock = cvzone.stackImages([imgStock, alert], 1, 1)

        #when face goes out of screen camera
        else:
            img = cv2.resize(img, (640, 360))
            default = np.zeros((512, 512, 3), dtype = "uint8")
            default = cv2.resize(default, (640, 360))
            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)
            cv2.putText(default, "Face Not Detected", (180, 180), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            imgStock=cvzone.stackImages([img, default], 2, 1)

        #display
        cv2.imshow("Blink Counter", imgStock)
        #press q to stop
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cv2.destroyAllWindows()distance (Distance Counter Algorithm)
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

def distanceCounter():
    cam = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    while True:
        success, img = cam.read()
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            # Drawing
            # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
            # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
            # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # # Finding the Focal Length
            # d = 50
            # f = (w*d)/W
            # print(f)

            # Finding distance
            f = 840
            d = (W * f) / w
            #print(d)

            alarm = np.zeros((512, 512, 3), dtype="uint8")
            alarm = cv2.resize(alarm, (640, 360))

            if d < 40:
                color = (255, 0, 255)
                alarm[:] = color
                cv2.putText(alarm, "Too Close", (250, 180), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            else:
                color = (0, 200, 0)
                alarm[:] = color
                cv2.putText(alarm, "Good Distance", (200, 180), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                               (face[10][0] - 120, face[10][1] - 70),
                               scale=2.5, colorR=color)

            img = cv2.resize(img, (640, 360))
            imgStock = cvzone.stackImages([img, alarm], 2, 1)

            cv2.imshow("Distance Counter", imgStock)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cv2.destroyAllWindows()
gameObject (Dynamic Object Eye-Exercise Game)
import cv2
import time
import numpy as np

img = np.zeros((512, 512, 3), dtype="uint8")

while True:
    cv2.putText(img, "<Stare at the object>", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1000)
    time.sleep(1)
    num = "3"
    cv2.putText(img, num, (250, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1000)
    time.sleep(1)

    num = "2"
    cv2.putText(img, num, (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Image", img)
    time.sleep(1)
    num = "1"
    cv2.putText(img, num, (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Image", img)
    time.sleep(1)
    num = " "
    time.sleep(1)

    rad = 10

    img = np.zeros((512, 512, 3), dtype="uint8")
    cv2.putText(img, "<Stare at the object>", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    rad+=10

    cv2.circle(img, (250, 250), rad, (0, 255, 0), -1)
    time.sleep(1)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
gameCursor (Follow the Cursor Eye-Exercise Game)
import cv2
import time
import numpy as np

def cursor():
    game = cv2.VideoCapture("assets/cursor.mov")
    
    img = np.zeros((512, 512, 3), dtype="uint8")
    
    rad = 20
    
    while True:
        cv2.putText(img, "<Stare at the object>", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1000)
        time.sleep(1)
        num = "3"
        cv2.putText(img, num, (250, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1000)
        time.sleep(1)
    
        num = "2"
        cv2.putText(img, num, (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img)
        time.sleep(1)
        num = "1"
        cv2.putText(img, num, (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img)
        time.sleep(1)
        num = " "
        time.sleep(1)
        
        while True:
            coor = (250, 0)
            cv2.circle(img, coor, rad, (0, 255, 0), -1)
            time.sleep(1)
    
            coor = (500, 250)
            cv2.circle(img, coor, rad, (0, 255, 0), -1)
            time.sleep(1)
    
            coor = (0, 500)
            cv2.circle(img, coor, rad, (0, 255, 0), -1)
            time.sleep(1)

            coor = (250, 500)
            cv2.circle(img, coor, rad, (0, 255, 0), -1)
            time.sleep(1)
            
            cv2.imshow("Image", img)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
