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
import os


# compare teach arrays on UI and total loading arrays, if loading arrays in teach arrays, save it
def teachListsDelCompare(teachLists, originTeachArrays):
    teachListsSplit = teachLists.replace(
        ' ......', '')  # delete " ......" on teachlists
    # split and ignore first element ("") on teachlists
    teachListsSplit = teachListsSplit.split("Delete ")[1:]
    originTeachArraysSplit = originTeachArrays.split("Delete ")[
        1:]  # split lists

    # check whether loading arrays in teach arrays, if yes, save to compareResult and return
    compareResult = []
    for originTeachArraySplit in originTeachArraysSplit:
        if originTeachArraySplit[:8] in teachLists:
            compareResult += [originTeachArraySplit]
    return compareResult


# save file
def saveFile(teachListsSplit):
    fileName = input("Please input the file name: ")
    if not os.path.isdir('database'):
        os.makedirs("database")

    with open('database/' + fileName, 'w') as file:
        for element in teachListsSplit:
            file.write(element + '\n')


def readFile(fileName):
    with open('database/' + fileName, 'r') as file:
        fls = file.read()
        # splited by \n, and ignore the space after the last list
        teachListsSplit = fls.split('\n')[:-1]
        return teachListsSplit


# parse validSpanValue and spectrumSliceValue from UI (for SVR use)
def SVRValidSpan():
    validSpanValue = soup.find_all(id='validSpanValue')[
        0].text  # get validSpanValue
    spectrumSliceValue = soup.find_all(id='spectrumSliceValue')[
        0].text  # get spectrumSliceValue
    return int(validSpanValue), int(spectrumSliceValue)


# alter row teach array to arraysFeature and arraysTarget, which is used on SVR ML
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


# feature scaling ((0.1,-0.1) to (1,-1))
def featureScaling(arraysFeature, arraysTarget):
    sc_X = StandardScaler()
#     sc_y = StandardScaler()
    X = sc_X.fit_transform(arraysFeature)
#     y    = sc_y.fit_transform(arraysTarget)
    y = arraysTarget
    return X, y


# split to train and test set (for SVR ML use)
def trainTestSplit(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=109)  # 70% training and 30% test
    return X_train, X_test, y_train, y_test


# fit and calculate accuracy
def generatingModel(X_train, X_test, y_train, y_test):
    regressor = svm.SVC(kernel='linear')     # Create a svm Classifier
    # Train the model using the training sets,ravel() covert to contiguous flattened array
    regressor.fit(X_train, y_train.ravel())
    # ex: [[1],[2]] to [1,2]
    y_pred = regressor.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return "{:.2f}".format(accuracy)


# try to train input file (on SVR model)
def tryInputFileTraining(X_train, y_train):
    try:
        regressor = svm.SVC(kernel='linear')         # Create a svm Classifier
        # Train the model using the training sets,ravel() convert to contiguous flattened array (ex: [[1],[2]] to [1,2])
        regressor.fit(X_train, y_train.ravel())
        return 'successful'
    except:
        return 'error'


# SVR predict new array (X_test)
def newArrayPredict(X_train, y_train, X_test):
    regressor = svm.SVC(kernel='linear')     # Create a svm Classifier
    # Train the model using the training sets,ravel() convert to contiguous flattened array (ex: [[1],[2]] to [1,2])
    regressor.fit(X_train, y_train.ravel())
    y_pred = regressor.predict(X_test)
    return y_pred


# cal result show on UI (teach mode)
def CalKeysToHtml(eventLogXPath):
    # find "resultLog" box on html
    result = driver.find_element(By.XPATH, eventLogXPath)
    result.send_keys("calculate successfully on teach mode! ")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


# accuracy result show on UI (teach mode)
def accuracyToHtml(accuracy, eventLogXPath):
    # find "resultLog" box on html
    result = driver.find_element(By.XPATH, eventLogXPath)
    result.send_keys("Accuracy: " + str(float(accuracy) * 100) + "%")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


# show training result (rel mode)
def LoadFileKeysToHtml(predictResult, eventLogXPath):
    # find "resultLog" box on html
    result = driver.find_element(By.XPATH, eventLogXPath)
    if predictResult == 'successful':
        result.send_keys(
            "Load file successfully, now you can start tapping test...")
    else:
        result.send_keys(
            "Oops! Something went wrong on SVR training, please check input data and try again")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


# show predict result (real mode)
def PredKeysToHtml(predResult, tapCount, eventLogXPath):
    result = driver.find_element(By.XPATH, eventLogXPath)
    result.send_keys(str(tapCount) + '. ' + str(predResult))
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


#####    parameter area      #####
teachDoneBool = False     # (teach mode) check whether teach done
validSpanValue = 320      # for test mode use, if html file can`t be detected
spectrumSliceValue = 10   # for test mode use, if html file can`t be detected
eventLogXPath = '/html/body/main/div/div[2]/div[2]/div/textarea'
oldTap = ''               # (real mode) check new array incoming
initFirst_bool = True     # (real mode) start to train mode
tapCount = 1              # (real mode) count how many new array was predicted

#####    parameter area end  #####


#####          Main          #####

# get file location
html_dir = input("Please enter the directory of html file: ")
driver = webdriver.Chrome()
driver.get('file:\\' + str(html_dir))
# driver.get('file:///C:/Users/flyboy/Google%20drive/to_company/html_recorder/project_test.html')

# for loop to get knocking array
while True:
    html = driver.page_source
    soup = BeautifulSoup(html)

    # teach and real mode get parameter
    result_log = soup.find_all(id='resultLog')[0].text

    # teach mode: select array list and start ML training
    if "Teach arrays are under calculating..." in result_log and teachDoneBool == False:
        # parse teach array from UI
        teachLists = soup.find_all(id='teachArrays')[0].text
        originTeachArrays = driver.execute_script(
            'return teachArraySend()')     # parse teach array from html
        # compare UI and html teach array, the same array will be saved
        teachListsSplit = teachListsDelCompare(teachLists, originTeachArrays)
        # save these arrays to file
        saveFile(teachListsSplit)
#         teachListsSplit                  = readFile('test20220122.txt')                         # read arrays file
        # parse SVR parameter from UI
        validSpanValue, spectrumSliceValue = SVRValidSpan()
        arraysFeature, arraysTarget = listsToArrays(
            teachListsSplit, oneArrayLen=validSpanValue, oneSegmentLen=spectrumSliceValue)
        # alter row teach array to arraysFeature and arraysTarget, which is used on SVR ML
        # feature scaling ((0.1,-0.1) to (1,-1))
        X, y = featureScaling(arraysFeature, arraysTarget)
        # split to train and test set (for SVR ML use)
        X_train, X_test, y_train, y_test = trainTestSplit(X, y, 0.3)
        # fit and calculate accuracy
        accuracy = generatingModel(X_train, X_test, y_train, y_test)
        # cal result show on UI
        CalKeysToHtml(eventLogXPath)
        # accuracy result show on UI
        accuracyToHtml(accuracy, eventLogXPath)
        teachDoneBool = True

    # Real mode: select file and start predict the sound
    if "Start initializing on real mode..." in result_log:
        # parse SVR parameter from UI
        validSpanValue, spectrumSliceValue = SVRValidSpan()

        # first file initializing
        # select file to train model
        if initFirst_bool == True:
            readFileInReal = soup.find_all(id='initializeButton')[0].text.replace(
                ' Initialize?', '')  # get input file name to train
            teachListsSplit = readFile(readFileInReal)
            # parse SVR parameter from UI
            validSpanValue, spectrumSliceValue = SVRValidSpan()
            arraysFeature, arraysTarget = listsToArrays(
                teachListsSplit, oneArrayLen=validSpanValue, oneSegmentLen=spectrumSliceValue)
            # alter row teach array to arraysFeature and arraysTarget, which is used on SVR ML
            # feature scaling ((0.1,-0.1) to (1,-1))
            X, y = featureScaling(arraysFeature, arraysTarget)
            # split to train and test set (for SVR ML use)
            X_train, X_test, y_train, y_test = trainTestSplit(X, y, 0.3)
            # try to train input file (on SVR model)
            predictResult = tryInputFileTraining(X_train, y_train)
            # show training result
            LoadFileKeysToHtml(predictResult, eventLogXPath)
            initFirst_bool = False

        # start to predict incoming new array
        else:
            # get new array to predict
            newTap = soup.find_all(id='newArray')[0].text
            # check new array is coming
            if newTap != oldTap:
                X_test = listToArray(newTap, oneArrayLen=validSpanValue,
                                     oneSegmentLen=spectrumSliceValue)  # alter str to 2D array
                # SVR predict new array (X_test)
                predResult = newArrayPredict(X_train, y_train, X_test)
                # show predict result
                PredKeysToHtml(predResult, tapCount, eventLogXPath)
                oldTap = newTap                                         # to check new tap coming
                tapCount += 1
