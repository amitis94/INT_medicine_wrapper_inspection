import time
from PIL import Image
import os
import numpy as np
from machine_vision.model.functions import GENERATE_filename, Preprocessing_BIG, Preprocessing_TINY, MEASUREMENTS, GET_golden_image
from machine_vision.schemas import InferenceResultModel
import cv2

golden_template_0 = GET_golden_image(0) # default size 1024, 1024
golden_template_1 = GET_golden_image(1) # default size 1024, 1024

def inference(save_dir: str, camera_index: int, image: Image, filename: str, golden_template_0=golden_template_0, golden_template_1=golden_template_1):
    try:
        start = time.time()
        image_pil_gray = image.convert('L')
        image_cv2_gray = np.array(image_pil_gray)
        
        if camera_index == 0:
            golden_template = golden_template_0
        else:
            golden_template = golden_template_1
        
        big_target_roi_img, big_target_preprocessed_img = Preprocessing_BIG(image_cv2_gray, camera_index, golden_template)
        tiny_target_roi_resize_img, tiny_target_preprocessed_img = Preprocessing_TINY(image_cv2_gray, camera_index) # 내부에서 camera index 받아서 처리함
        bad_count, result_image, defect_locations = MEASUREMENTS(big_target_roi_img, big_target_preprocessed_img, tiny_target_roi_resize_img, tiny_target_preprocessed_img)
        
        os.makedirs(save_dir+ "/good/", exist_ok=True)
        os.makedirs(save_dir + "/bad/", exist_ok=True)
        raw_name = f"{save_dir}/{filename}"
        
        if bad_count > 0:
            print("too big object")
            raw_name = f"{save_dir}/bad/{filename}"
            #result_name = f"{save_dir}/bad/{GENERATE_filename(camera_index, is_result=True)}"
            result_name = raw_name
            cv2.imwrite(result_name, result_image)
            end = time.time()
            #cv2.imwrite(result_name, result_image)
            print(f"{end - start:.5f} sec")
            
            infer_result = InferenceResultModel(
                is_success = True,
                error_message = "",
                inspection_result = "bad",
                bad_object_count = bad_count,
                raw_image_path = raw_name,
                result_image_path = result_name,
                defects = defect_locations  #todo : 보국씨 defects 에 list[DefectModel] 타입으로 반영 바랍니다.
            )
            return infer_result
            #return {"inspection_result": "bad", "bad_object_count":bad_count,"raw_image_path": raw_name, "result_image_path": result_name}
        else:
            raw_name = f"{save_dir}/good/{filename}"
            result_name = f"{save_dir}/good/{GENERATE_filename(camera_index, is_result=True)}"
            cv2.imwrite(result_name, result_image)
            end = time.time()
            #cv2.imwrite(result_name, result_image)
            print(f"{end - start:.5f} sec")
            
            infer_result = InferenceResultModel(
                is_success = True,
                error_message = "",
                inspection_result = "good",
                bad_object_count = 0,
                raw_image_path = raw_name,
                result_image_path = result_name,
                defects = []  
            )
            return infer_result
            #return {"inspection_result": "good", "bad_object_count":0, "raw_image_path": raw_name, "result_image_path": result_name}
    except Exception as e:
        raise