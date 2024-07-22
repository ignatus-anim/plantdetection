# from typing import Annotated

# from fastapi import FastAPI, File, Form, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import json

# app = FastAPI()

# origins = [
#     "http://localhost:8081",
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Load the model when the app starts

# # Read solutions data from JSON


# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"


# def read_file_as_image(data) -> np.ndarray:
#     image = Image.open(BytesIO(data)).resize((128, 128))
#     image = np.array(image)
#     if image.shape[-1] == 4:  # Check if the image has an alpha channel
#         image = image[..., :3]  # Remove the alpha channel
#     return image


# @app.post("/predict")
# async def predict(crop: Annotated[str, Form()], file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)

#     with open(f"solutions/{crop.lower()}_solution.json", "r") as solutions_file:
#         SOLUTIONS = json.load(solutions_file)
#     MODEL = tf.keras.models.load_model(f"models/{crop.lower()}.keras")

#     predictions = MODEL.predict(img_batch)
#     index = np.argmax(predictions[0])

#     disease_info = SOLUTIONS[index]

#     return {
#         "class": disease_info["name"],
#         "confidence": float(np.max(predictions[0])),
#         "causes": disease_info["causes"],
#         "recommended_solutions": disease_info["recommended_solutions"],
#     }


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)




from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

app = FastAPI()

origins = [
    "http://localhost:8081",
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((128, 128))
    image = np.array(image)
    if image.shape[-1] == 4:  # Check if the image has an alpha channel
        image = image[..., :3]  # Remove the alpha channel
    return image

def load_model_and_solution(crop):
    model_file = f"models/{crop}.tflite"
    solution_file = f"solutions/{crop}_solution.json"

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # Load crop diseases solutions
    with open(solution_file, 'r') as solutions:
        crop_diseases = json.load(solutions)

    return interpreter, crop_diseases

def model_prediction(image, interpreter):
    input_arr = np.expand_dims(image, axis=0).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    result_index = np.argmax(prediction)
    return result_index, prediction

@app.post("/predict")
async def predict(crop: Annotated[str, Form()], file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    interpreter, crop_diseases = load_model_and_solution(crop.lower())
    result_index, prediction = model_prediction(image, interpreter)
    confidence = round((float(np.max(prediction)) * 100), 2)

    if confidence < 80:
        return {"error": "Confidence is below 80%. Please upload a clear image."}

    disease_info = crop_diseases[result_index]

    return {
        "class": disease_info["name"],
        "confidence": confidence,
        "causes": disease_info["causes"],
        "recommended_solutions": disease_info["recommended_solutions"],
        "recommended_pesticide": disease_info["recommended_pesticide"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
