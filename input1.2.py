# Selenium get web
from bs4 import BeautifulSoup
from selenium import webdriver
from pathlib import Path
import time
from statistics import mean
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


# split lists by string "Delete(ongoing)"
def listsSplit(teachLists):
    # split each list by string "Delete(ongoing)"
    teachListsSplit = teachLists.split('Delete(ongoing)')
    # delete first element which is empty element( '' )
    teachListsSplit = teachListsSplit[1:]
    return teachListsSplit

# save file


def saveFile(teachListsSplit, fileName):
    with open('database/' + fileName, 'w') as file:
        for element in teachListsSplit:
            file.write(element + '\n')


def readFile(fileName):
    with open('database/' + fileName, 'r') as file:
        fls = file.read()
        # splited by \n, and ignore the space after the last list
        teachListsSplit = fls.split('\n')[:-1]
        return teachListsSplit

# list data to array


def listsToArrays(teachListsSplit, oneArrayLen, oneSegmentLen):
    listsFeature = []
    listsTarget = []
    for teachList in teachListsSplit:
        listSplit = teachList.split(",")
        pureList = listSplit[2:]
        listSegment = []
        for i in range(0, int(oneArrayLen/oneSegmentLen)):
            # convert str list to int list
            int_list = list(
                map(int, pureList[i * oneSegmentLen: (i+1) * oneSegmentLen]))
            # convert float to int, and save to an 2D list
            listSegment += [int(mean(int_list))]
        listsFeature += [listSegment]           # get feature
        listsTarget += [int(listSplit[1])]    # get target

    arraysFeature = np.array(listsFeature)       # list to array
    arraysTarget = np.array(listsTarget)        # list to array
    arraysTarget = arraysTarget.reshape(
        len(arraysTarget), 1)  # 1D array to 2D array
    return arraysFeature, arraysTarget

# str to 2D array


def listToArray(newList, oneArrayLen, oneSegmentLen):
    listsFeature = []
    listsTarget = []
    listSplit = newList.split(",")
    listSegment = []
    for i in range(0, int(oneArrayLen/oneSegmentLen)):
        # convert str list to int list
        int_list = list(
            map(int, listSplit[i * oneSegmentLen: (i+1) * oneSegmentLen]))
        # convert float to int, and save to an 2D list
        listSegment += [int(mean(int_list))]
    listsFeature = listSegment           # get feature
#     print(listsFeature)
    arrayFeature = np.array([listsFeature])       # list to array
    return arrayFeature


def featureScaling(arraysFeature, arraysTarget):
    sc_X = StandardScaler()
#     sc_y = StandardScaler()
    X = sc_X.fit_transform(arraysFeature)
#     y    = sc_y.fit_transform(arraysTarget)
    y = arraysTarget
    return X, y


def trainTestSplit(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=109)  # 70% training and 30% test
    return X_train, X_test, y_train, y_test


def generatingModel(X_train, X_test, y_train, y_test):
    regressor = svm.SVC(kernel='linear')     # Create a svm Classifier
    # Train the model using the training sets,ravel() covert to contiguous flattened array
    regressor.fit(X_train, y_train.ravel())
    # ex: [[1],[2]] to [1,2]
    y_pred = regressor.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return "{:.2f}".format(accuracy)


def trainAndTest(X_train, y_train, X_test, initFirst_bool, realX_testBool):
    if initFirst_bool == True:
        regressor = svm.SVC(kernel='linear')     # Create a svm Classifier
        # Train the model using the training sets,ravel() covert to contiguous flattened array
        regressor.fit(X_train, y_train.ravel())
        # ex: [[1],[2]] to [1,2]
        return 'train mode'
    else:
        if realX_testBool == True:
            regressor = svm.SVC(kernel='linear')     # Create a svm Classifier
            # Train the model using the training sets,ravel() covert to contiguous flattened array
            regressor.fit(X_train, y_train.ravel())
            # ex: [[1],[2]] to [1,2]
            y_pred = regressor.predict(X_test)
            return y_pred
        else:
            return 0


def CalKeysToHtml():
    result = driver.find_element(By.XPATH, '/html/body/div[6]/textarea')
    result.send_keys("calculate successfully on teach mode! ")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


def accuracyToHtml(accuracy):
    result = driver.find_element(By.XPATH, '/html/body/div[6]/textarea')
    result.send_keys("Accuracy: " + str(float(accuracy) * 100) + "%")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


def LoadFileKeysToHtml():
    result = driver.find_element(By.XPATH, '/html/body/div[6]/textarea')
    result.send_keys(
        "Load file successfully, now you can start tapping test...")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


def PredKeysToHtml(predResult, tapCount):
    result = driver.find_element(By.XPATH, '/html/body/div[6]/textarea')
    result.send_keys(str(tapCount) + '. ' + str(predResult))
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


############# Main ############

html_dir = input("Please enter the directory of html file: ")
oneArrayLen = input("Please enter the array total length (default=1024): ")
oneArrayLen = int(oneArrayLen)
oneSegmentLen = input("Please enter the Segment length (default=32): ")
oneSegmentLen = int(oneSegmentLen)


driver = webdriver.Chrome()
# driver.get('http://127.0.0.1:5500/project_test.html')

# select this director
# html_dir = Path(__file__).resolve().parent.joinpath('project_test.html')
driver.get('file:\\' + str(html_dir))
# driver.get('file:///C:/Users/flyboy/Google%20drive/to_company/html_recorder/project_test.html')

# oneArrayLen = 1024
# oneSegmentLen = 32
oldTap = ''
initFirst_bool = True
realX_testBool = False
tapCount = 1

# for loop to get knocking array
while True:
    html = driver.page_source
    soup = BeautifulSoup(html)
    readFileInReal = soup.find_all(id='initializeButton')[
        0].text.replace(' Initialize?', '')
    cal_bool = soup.find_all(id='Calculating')[0].text
    initFile_bool = soup.find_all(id='initialize')[0].text
    if cal_bool == "Calculating...":
        teachLists = soup.find_all(id='teachArrays')[0].text
        teachListsSplit = listsSplit(teachLists)
        file_name = input("Please input file name: ")
        saveFile(teachListsSplit, file_name)
        # teachListsSplit                  = readFile('test20220122.txt')
        arraysFeature, arraysTarget = listsToArrays(
            teachListsSplit, oneArrayLen=1024, oneSegmentLen=32)
        X, y = featureScaling(arraysFeature, arraysTarget)
        X_train, X_test, y_train, y_test = trainTestSplit(X, y, 0.3)
        accuracy = generatingModel(X_train, X_test, y_train, y_test)
        CalKeysToHtml()
        accuracyToHtml(accuracy)
        break
    if initFile_bool == "Start initializing...":
        if initFirst_bool == True:
            #             fileName + ' <button id="startInitialize">Initialize?</button>'
            teachListsSplit = readFile(readFileInReal)
            arraysFeature, arraysTarget = listsToArrays(
                teachListsSplit, oneArrayLen=oneArrayLen, oneSegmentLen=oneSegmentLen)
            X, y = featureScaling(arraysFeature, arraysTarget)
            X_train, X_test, y_train, y_test = trainTestSplit(X, y, 0.3)
        else:
            newTap = soup.find_all(id='newArray')[0].text
            if newTap != oldTap:
                X_test = listToArray(
                    newTap, oneArrayLen=oneArrayLen, oneSegmentLen=oneSegmentLen)
                realX_testBool = True
                oldTap = newTap

        predResult = trainAndTest(
            X_train, y_train, X_test, initFirst_bool, realX_testBool)
        if initFirst_bool == True:
            LoadFileKeysToHtml()
        else:
            if realX_testBool == True:
                PredKeysToHtml(predResult, tapCount)
                tapCount += 1

        initFirst_bool = False
        realX_testBool = False
