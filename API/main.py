from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
MODEL=tf.keras.models.load_model("C:\code\colon\models\masked")
MODEL2=tf.keras.models.load_model("C:\code\colon\models\15_epoch_lung")
CLASS_NAMES=['colon_aca', 'colon_n']
CLASS_NAMES2=['lung_aca', 'lung_n', 'lung_scc']
@app.get("/first")
async def check():
    return "its working"
def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image
@app.post("/pic")
async def predict(
        file: UploadFile = File(...)
):
    image =read_file_as_image(await file.read())
    batch1=np.expand_dims(image,0)
    prediction=MODEL.predict(batch1)
    pred_dis=CLASS_NAMES[np.argmax(prediction[0])]
    conf=np.max(prediction[0])
    return{
        'class':pred_dis,
        'confidence':float(conf)
    }
@app.post("/pic")
async def predict(
        file: UploadFile = File(...)
):
    image =read_file_as_image(await file.read())
    batch1=np.expand_dims(image,0)
    prediction=MODEL2.predict(batch1)
    pred_dis=CLASS_NAMES2[np.argmax(prediction[0])]
    conf=np.max(prediction[0])
    return{
        'class':pred_dis,
        'confidence':float(conf)
    }
if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8008)