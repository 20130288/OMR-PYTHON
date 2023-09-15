import cv2 as cv
import numpy as np
import utils as ut
import glob
import pandas as pd
import openpyxl

# DECLARE VARIABLE
widthImg = 800
heightImg = 1000
widthImgCode = 300
heightImgCode = 1000
question = 5
choice = 4
idRow = 10
idCol = 8
codeRow = 10
codeCol = 3
answer001 = [1, 1, 3, 4, 2]
answer112 = [1, 2, 1, 3, 4]
answer113 = [3, 4, 1, 2, 2]
id_list = []
code_list = []
score_list = []
#############################
folder_dir = "Y:/AI/OMR/img"
for image in glob.iglob(f'{folder_dir}/*'):
    if (image.endswith(".png") or image.endswith(".jpg")):

        # CONVERT IMAGE
        img = cv.imread(image)
        img = cv.resize(img, (widthImg, heightImg))
        imgContours = img.copy()
        imgCoinerPoint = img.copy()
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv.Canny(imgBlur, 10, 50)

        # FIND ALL CONTOURS
        contours, hierachy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

        # FIND RECT
        rectContour = ut.rectContours(contours)
        choiceRect = ut.getConnerPoint(rectContour[1])
        idRect = ut.getConnerPoint(rectContour[0])
        codeRect = ut.getConnerPoint(rectContour[2])

        if choiceRect.size != 0 and idRect.size != 0:
            cv.drawContours(imgCoinerPoint, choiceRect, -1, (0, 0, 255), 10)
            cv.drawContours(imgCoinerPoint, idRect, -1, (255, 0, 0), 10)
            cv.drawContours(imgCoinerPoint, codeRect, -1, (255, 255, 204), 10)

        # WARP AND CROP IMAGE
        choiceRect = ut.reOrderPoint(choiceRect)
        idRect = ut.reOrderPoint(idRect)
        codeRect = ut.reOrderPoint(codeRect)

        ptChoice1 = np.float32(choiceRect)
        ptChoice2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrixChoice = cv.getPerspectiveTransform(ptChoice1, ptChoice2)
        imgWarpedChoice = cv.warpPerspective(img, matrixChoice, (widthImg, heightImg))

        ptId1 = np.float32(idRect)
        ptId2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrixId = cv.getPerspectiveTransform(ptId1, ptId2)
        imgWarpedId = cv.warpPerspective(img, matrixId, (widthImg, heightImg))

        ptCode1 = np.float32(codeRect)
        ptCode2 = np.float32([[0, 0], [widthImgCode, 0], [0, heightImgCode], [widthImgCode, heightImgCode]])
        matrixCode = cv.getPerspectiveTransform(ptCode1, ptCode2)
        imgWarpedCode = cv.warpPerspective(img, matrixCode, (widthImgCode, heightImgCode))

        # APPLY THRESHOLD
        imgWarpedChoiceGray = cv.cvtColor(imgWarpedChoice, cv.COLOR_BGR2GRAY)
        imgWarpedChoiceThres = cv.threshold(imgWarpedChoiceGray, 160, 255, cv.THRESH_BINARY_INV)[1]

        imgWarpedIdGray = cv.cvtColor(imgWarpedId, cv.COLOR_BGR2GRAY)
        imgWarpedIdThres = cv.threshold(imgWarpedIdGray, 150, 255, cv.THRESH_BINARY_INV)[1]

        imgWarpedCodeGray = cv.cvtColor(imgWarpedCode, cv.COLOR_BGR2GRAY)
        imgWarpedCodeThres = cv.threshold(imgWarpedCodeGray, 150, 255, cv.THRESH_BINARY_INV)[1]

        # SPLIT CHOICE BOXES
        boxes = ut.splitChoiceBoxes(imgWarpedChoiceThres)

        pixelVal = np.zeros((question, choice))
        countR = 0
        countC = 0
        for box in boxes:
            totalPixel = cv.countNonZero(box)
            pixelVal[countR][countC] = totalPixel
            countC += 1
            if countC == choice:
                countR += 1
                countC = 0

        # SPLIT ID
        idBoxes = ut.splitIdBoxes(imgWarpedIdThres)

        pixelVal_Id = np.zeros((idRow, idCol))
        countRId = 0
        countCId = 0
        for box in idBoxes:
            totalPixel = cv.countNonZero(box)
            pixelVal_Id[countRId][countCId] = totalPixel
            countRId += 1
            if countRId == idRow:
                countCId += 1
                countRId = 0
        pixel_Id_Transpose = pixelVal_Id.transpose()

        # SPLIT CODE
        codeBoxes = ut.splitCodeBoxes(imgWarpedCodeThres)

        pixelVal_Code = np.zeros((codeRow, codeCol))
        countRCode = 0
        countCCode = 0
        for box in codeBoxes:
            totalPixel = cv.countNonZero(box)
            pixelVal_Code[countRCode][countCCode] = totalPixel
            countRCode += 1
            if (countRCode == codeRow):
                countCCode += 1
                countRCode = 0
        pixelVal_Code_Transpose = pixelVal_Code.transpose()

        # FIND CHOICE ID
        myChoiceId = []

        for i in range(0, idCol):
            array = pixel_Id_Transpose[i]
            indexVal = np.where(array == np.amax(array))
            if np.amax(array) > 4900:
                myChoiceId.append(indexVal[0][0])
            else:
                myChoiceId.append('E')
        studentId = ""
        for i in range(0, 8):
            studentId += str(myChoiceId[i])
        print("Student: ", studentId)

        # FIND CODE
        myCode = []

        for i in range(0, codeCol):
            array = pixelVal_Code_Transpose[i]
            indexVal = np.where(array == np.amax(array))
            if np.amax(array) > 4900:
                myCode.append(indexVal[0][0])
            else:
                myCode.append('E')
        code = ""
        for i in range(0, codeCol):
            code += str(myCode[i])
        print("Ma De: ", code)

        # FIND MY CHOICE
        mychoice = []

        for i in range(0, question):
            array = pixelVal[i]
            indexVal = np.where(array == np.amax(array))
            if np.amax(array) > 13000:
                mychoice.append(indexVal[0][0] + 1)
            else:
                mychoice.append(0)
        print("myChoice: ", mychoice)

        # CHECK CHOICE
        grading = []

        for i in range(0, question):
            if code == "001":
                print(answer001)
                if mychoice[i] == answer001[i]:
                    grading.append(1)
                else:
                    grading.append(0)
            if code == "112":
                print(answer112)
                if mychoice[i] == answer112[i]:
                    grading.append(1)
                else:
                    grading.append(0)
            if code == "113":
                print(answer113)
                if mychoice[i] == answer113[i]:
                    grading.append(1)
                else:
                    grading.append(0)

        # GET SCORE
        score = sum(grading) / question * 10
        print("Score: ", score)

        # APPEND ALL VALUE TO LIST
        id_list.append(studentId)

        code_list.append(code)

        score_list.append(score)

# WRITE DATA TO EXCEL
data = {"MSSV": id_list, "Mã đề": code_list, "Điểm": score_list}
df = pd.DataFrame(data)
df_final = df.sort_values(by=['MSSV'])
df_final.to_excel('Y:/AI/OMR/result/result.xlsx', index=False, header=True)
