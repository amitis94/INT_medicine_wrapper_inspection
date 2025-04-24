from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Form
import uvicorn
import time
from PIL import Image
import io
import os
import datetime
from machine_vision.schemas import InferenceRequest, InferenceResultModel
import mmap
from machine_vision.main import inference
import sys

# FastAPI 인스턴스 생성
app = FastAPI()

@app.post("/inference/test", status_code=200)
def inference_test(response : Response, req : InferenceRequest):
    print(req.filename)
    try:
        image = Image.open(req.filename)
        basename = os.path.basename(req.filename)
        result = inference(save_dir=req.save_dir, camera_index=req.camera_index, image=image, filename=basename)
        return result
    except Exception as ex:
        print(ex)
        response.status_code = 500
        return { "is_success" : False, "error_message" : str(ex), "raw_image_path" : "", "result_image_path" : "" }

@app.post("/inference/test_mmf", status_code=200)
async def inference_test_mmf(response : Response, req : InferenceRequest):
    try:
        return await call_inference(req)
        #return await save_raw_image(req)
    except Exception as ex:
        response.status_code = 500
        print(ex)
        return { "is_success" : False, "error_message" : str(ex), "raw_image_path" : "", "result_image_path" : ""  }    

async def call_inference(req : InferenceRequest):
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    print(req.filename)
    start = time.time()
    try:
        print(req.filesize)
        basename = os.path.basename(req.filename)
        with mmap.mmap(-1, req.filesize, tagname=basename) as mm:
            image_data = mm[:req.filesize]
        mm.close()
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        #image.save(req.filename) #임시 원본 이미지 저장
        result = inference(save_dir=req.save_dir, camera_index=req.camera_index, image=image, filename=basename)
        end = time.time()
        print(f"{end - start:.5f} sec")
        print(datetime.datetime.now().strftime("%H:%M:%S"))
        return result
    except Exception as ex:
        print(ex)
        return InferenceResultModel(is_success=False, error_message= str(ex))

#원본이미지 취득을 위한 임시 함수 20250228
async def save_raw_image(req : InferenceRequest):
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    print(req.filename)
    start = time.time()
    try:
        print(req.filesize)
        basename = os.path.basename(req.filename)
        with mmap.mmap(-1, req.filesize, tagname=basename) as mm:
            image_data = mm[:req.filesize]
        mm.close()
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        image.save(req.filename)
        end = time.time()
        print(f"{end - start:.5f} sec")
        result = InferenceResultModel(
            is_success = True,
            error_message = "",
            inspection_result = "good",
            bad_object_count = 0,
            raw_image_path = req.filename,
            result_image_path = req.filename,
            defects = [])
        end = time.time()
        print(f"{end - start:.5f} sec")
        print(datetime.datetime.now().strftime("%H:%M:%S"))
        return result
    except Exception as ex:
        print(ex)
        return InferenceResultModel(is_success=False, error_message= str(ex))

@app.get("/check")    
def check_alive():
    current_time = datetime.datetime.now()
    print(current_time.strftime("%H:%M:%S"), f"{current_time.microsecond // 1000}ms")    

def check_path():
    print(sys.path) 
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, timeout_keep_alive=60000, workers=5) # Prod Env
    #uvicorn.run("server:app", host="0.0.0.0", port=8001, timeout_keep_alive=60000, reload=True) # Dev Env