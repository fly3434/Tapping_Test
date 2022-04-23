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
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential

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
def SVCGeneratingModel(X_train, X_test, y_train, y_test):
    regressor = svm.SVC(kernel='linear')     # Create a svm Classifier
    # Train the model using the training sets,ravel() covert to contiguous flattened array
    regressor.fit(X_train, y_train.ravel())
    # ex: [[1],[2]] to [1,2]
    y_pred = regressor.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return "{:.2f}".format(accuracy)


# reshape array for CNN use
def dataReshape(X):
    return X.reshape(-1, 18, 18, 1)


# build CNN model
def CNNGeneratingModel(X_train, X_test, y_train, y_test):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=15,  # randomly rotate images in the range 5 degrees
        zoom_range=0.01,  # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=8, kernel_size=5,
                                   activation='relu', input_shape=(18, 18, 1)))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    cnn.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    epochs = 50
    batch_size = 1

    # training
    history = cnn.fit_generator(datagen.flow(X_train,
                                             y_train,
                                             batch_size=batch_size),
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                validation_steps=X_test.shape[0] // batch_size)
    return cnn, 'successful'


# CNN prediction and get binary output
def CNNPredict(cnn, X_test):
    y_pred = cnn.predict(X_test)
    y_binary_pred = []
    for y in y_pred:
        if y > 0.5:
            y_binary_pred.append(1)
        else:
            y_binary_pred.append(0)
    return y_binary_pred


# get CNN accuracy
def CNNAcc(y_test, cnn_y_binary_pred):
    return accuracy_score(y_test, cnn_y_binary_pred)


# try to train input file (on SVR model)
def trySVCTraining(X_train, y_train):
    try:
        global regressor
        regressor = svm.SVC(kernel='linear')         # Create a svm Classifier
        # Train the model using the training sets,ravel() convert to contiguous flattened array (ex: [[1],[2]] to [1,2])
        regressor.fit(X_train, y_train.ravel())
        return 'successful'
    except:
        return 'error'


# cal result show on UI (teach mode)
def CalKeysToHtml(eventLogXPath):
    # find "resultLog" box on html
    result = driver.find_element(By.XPATH, eventLogXPath)
    result.send_keys("calculate successfully on teach mode! ")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


# accuracy result show on UI (teach mode)
def accuracyToHtml(SVCAccuracy, CNNAccuracy, eventLogXPath):
    # find "resultLog" box on html
    result = driver.find_element(By.XPATH, eventLogXPath)
    result.send_keys("SVC accuracy: " + str(float(SVCAccuracy) * 100) + "%")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()
    result.send_keys("CNN accuracy: " + str(float(CNNAccuracy) * 100) + "%")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()

# show training result (rel mode)


def LoadFileKeysToHtml(SVCPredictResult, CNNPredictResult, eventLogXPath):
    # find "resultLog" box on html
    result = driver.find_element(By.XPATH, eventLogXPath)
    if SVCPredictResult == 'successful' and CNNPredictResult == 'successful':
        result.send_keys(
            "Load file successfully, now you can start tapping test...")
    else:
        result.send_keys(
            "Oops! Something went wrong on SVR or CNN training, please check input data and try again")
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


# show predict result (real mode)
def PredKeysToHtml(SVCPredResult, CNNPredResult, tapCount, eventLogXPath):
    result = driver.find_element(By.XPATH, eventLogXPath)
    result.send_keys(str(tapCount) + '. SVC: ' +
                     str(SVCPredResult) + ' CNN: ' + str(CNNPredResult))
    ActionChains(driver).key_down(Keys.SHIFT).key_down(
        Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()


#####    parameter area   #####

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
    html         = driver.page_source
    soup         = BeautifulSoup(html)
    
    # teach and real mode get parameter
    result_log   = soup.find_all(id='resultLog')[0].text
    
    # teach mode: select array list and start ML training
    if "Teach arrays are under calculating..." in result_log and teachDoneBool == False:
        teachLists                         = soup.find_all(id='teachArrays')[0].text              # parse teach array from UI
        originTeachArrays                  = driver.execute_script('return teachArraySend()')     # parse teach array from html
        teachListsSplit                    = teachListsDelCompare(teachLists, originTeachArrays)  # compare UI and html teach array, the same array will be saved
        saveFile(teachListsSplit)                                                                 # save these arrays to file
#         teachListsSplit                  = readFile('test20220122.txt')                         # read arrays file
        
        # SVR 
        validSpanValue, spectrumSliceValue = SVRValidSpan()                                       # parse SVR parameter from UI
        SVCarraysFeature, SVCarraysTarget  = listsToArrays(teachListsSplit, oneArrayLen=validSpanValue, oneSegmentLen=spectrumSliceValue)
        # alter row teach array to arraysFeature and arraysTarget, which is used on SVR ML
        SVC_X, SVC_y                       = featureScaling(SVCarraysFeature, SVCarraysTarget)    # feature scaling ((0.1,-0.1) to (1,-1))
        S_X_train, S_X_test, S_y_train, S_y_test   = trainTestSplit(SVC_X, SVC_y, 0.3)                       # split to train and test set (for SVR ML use)
        SVCAccuracy                        = SVCGeneratingModel(S_X_train, S_X_test, S_y_train, S_y_test)    # fit and calculate accuracy
        
        # CNN
        CNNarraysFeature, CNNarraysTarget  = listsToArrays(teachListsSplit, oneArrayLen=324, oneSegmentLen=1) 
        # alter row teach array to arraysFeature and arraysTarget, which is used on SVR ML
        SVC_X, SVC_y                       = featureScaling(CNNarraysFeature, CNNarraysTarget)          # feature scaling ((0.1,-0.1) to (1,-1))
        X_reshape                          = dataReshape(SVC_X)
        C_X_train, C_X_test, C_y_train, C_y_test   = trainTestSplit(X_reshape, SVC_y, 0.3)
        cnn, CNNPredictResult              = CNNGeneratingModel(C_X_train, C_X_test, C_y_train, C_y_test)
        cnn_y_binary_pred                  = CNNPredict(cnn, C_X_test)
        CNNAccuracy                        = CNNAcc(C_y_test, cnn_y_binary_pred)
        
        CalKeysToHtml(eventLogXPath)                                                              # cal result show on UI
        accuracyToHtml(SVCAccuracy, CNNAccuracy, eventLogXPath)                                   # accuracy result show on UI
        teachDoneBool                      = True
        
    # Real mode: select file and start predict the sound
    if "Start initializing on real mode..." in result_log:
#         validSpanValue, spectrumSliceValue     = SVRValidSpan()                                 # parse SVR parameter from UI
        
        # first file initializing
        if initFirst_bool == True:                                                              # select file to train model
            readFileInReal                     = soup.find_all(id='initializeButton')[0].text.replace(' Initialize?', '') # get input file name to train
            teachListsSplit                    = readFile(readFileInReal)
            
            # SVC
            validSpanValue, spectrumSliceValue = SVRValidSpan()                                 # parse SVR parameter from UI
            SVCarraysFeature, SVCarraysTarget  = listsToArrays(teachListsSplit, oneArrayLen=validSpanValue, oneSegmentLen=spectrumSliceValue)
            # alter row teach array to arraysFeature and arraysTarget, which is used on SVR ML
            SVC_X, SVC_y                       = featureScaling(SVCarraysFeature, SVCarraysTarget)    # feature scaling ((0.1,-0.1) to (1,-1))
            S_X_train, S_X_test, S_y_train, S_y_test   = trainTestSplit(SVC_X, SVC_y, 0.3)                      # split to train and test set (for SVR ML use)
            SVCPredictResult                   = trySVCTraining(S_X_train, S_y_train)         # try to train input file (on SVR model)
            
            # CNN
            CNNarraysFeature, CNNarraysTarget  = listsToArrays(teachListsSplit, oneArrayLen=324, oneSegmentLen=1) 
            # alter row teach array to arraysFeature and arraysTarget, which is used on SVR ML
            SVC_X, SVC_y                       = featureScaling(CNNarraysFeature, CNNarraysTarget)          # feature scaling ((0.1,-0.1) to (1,-1))
            X_reshape                          = dataReshape(SVC_X)
            C_X_train, C_X_test, C_y_train, C_y_test   = trainTestSplit(X_reshape, SVC_y, 0.3)
            cnn, CNNPredictResult              = CNNGeneratingModel(C_X_train, C_X_test, C_y_train, C_y_test)
            cnn_y_binary_pred                  = CNNPredict(cnn, C_X_test)
            CNNAccuracy                        = CNNAcc(C_y_test, cnn_y_binary_pred)
                
            
            LoadFileKeysToHtml(SVCPredictResult, CNNPredictResult, eventLogXPath)                                    # show training result
            initFirst_bool                     = False
            
        # start to predict incoming new array
        else:
            newTap                                 = soup.find_all(id='newArray')[0].text           # get new array to predict
            if newTap != oldTap:                                                                # check new array is coming
                # SVC
                SVC_X_new                          = listToArray(newTap, oneArrayLen=validSpanValue, oneSegmentLen=spectrumSliceValue) # alter str to 2D array
                SVCPredResult                      = regressor.predict(SVC_X_new)      # SVR predict new array (X_test)
                
                # CNN
                CNN_X_new                          = listToArray(newTap, oneArrayLen=324, oneSegmentLen=1) # alter str to 2D array
                CNN_X_new                          = CNN_X_new.reshape(1, 18, 18, 1)
                CNNPredResult                      = CNNPredict(cnn, CNN_X_new)
                
                PredKeysToHtml(SVCPredResult, CNNPredResult, tapCount, eventLogXPath)                             # show predict result
                oldTap                             = newTap                                         # to check new tap coming
                tapCount += 1
