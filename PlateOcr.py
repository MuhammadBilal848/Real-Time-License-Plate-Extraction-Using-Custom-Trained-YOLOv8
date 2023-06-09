import cv2 as cv
import keras_ocr
import os
import numpy as np
import pandas as pd



def keras_ocr_on_numberplate():
    folder_path = './LicensePlate'  # Replace with the actual folder path

    # Get the list of all files in the folder
    file_names = os.listdir(folder_path)

    # Filter out directories, if any
    file_names = [file for file in file_names if os.path.isfile(os.path.join(folder_path, file))]

    file_names_with_prefix = ['LicensePlate/' + file for file in file_names]
    file_len = len(file_names_with_prefix)

    print(file_len)
    print(file_names_with_prefix)


    ppl = keras_ocr.pipeline.Pipeline()

    result = []

    prediction = [keras_ocr.tools.read(img) for img in file_names_with_prefix]
    image = ppl.recognize(prediction)

    df = pd.DataFrame(image)

    for i in range(file_len):

        # checks if one of the column is empty
        if ((df.iloc[:, 0])[i] == None) or ((df.iloc[:, 1])[i] == None):
            if ((df.iloc[:, 0])[i] == None):
                result.append({'Image': file_names_with_prefix[i], 'License Plate': ((df.iloc[:, 1])[i][0]).upper()})
            else:
                result.append({'Image': file_names_with_prefix[i], 'License Plate': ((df.iloc[:, 0])[i][0]).upper()})
            continue

        # checks if both column 1 and column 2 are empty
        if ((df.iloc[:, 0])[i] == None) and ((df.iloc[:, 1])[i] == None):
            print('License Plate Could Not detected')  

        result.append({'Image': file_names_with_prefix[i], 'License Plate': ((df.iloc[:, 0])[i][0] + (df.iloc[:, 1])[i][0]).upper()})

    df = pd.DataFrame(result)
    df.to_csv('License Plates.csv')

keras_ocr_on_numberplate()
