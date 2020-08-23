from fastapi import FastAPI, File, UploadFile


from typing import Optional
import numpy as np
from PIL import Image
import operator


from tensorflow.keras.models import load_model
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model = load_model(os.path.join(this_dir, "inception_cnn.h5"))

app = FastAPI()


monkey_label = ["Mantled Howler", "Patas Monkey", "Bald Uakari", "Japanese Macaque", "Pygmy Marmoset",
                "White headed Capuchin", "Silvery Marmoset", "Common Squirrel Monkey", "Black Headed Night Monkey", "Nilgiri Langur"]


@app.post("/predict/")
async def findMonkeySpecies(Uploadedfile: UploadFile = File(...)):

    try:
        img = Image.open(Uploadedfile.file)
        img = img.resize((299, 299))
        img = np.array(img)
        img = img/255
        img = img.reshape(1, 299, 299, 3)

        predictions = cnn_model.predict(img)
        predictions = predictions.round(1)
        print("########", predictions[0])
        output = predictions[0]
    except:
        return {"error": "ERROR EXTRACTING THE FILE"}
    index, value = max(enumerate(output), key=operator.itemgetter(1))

    return {"prediction": monkey_label[index]}
