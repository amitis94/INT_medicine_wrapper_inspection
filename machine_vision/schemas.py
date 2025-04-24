from pydantic import BaseModel

class InferenceRequest(BaseModel):
    save_dir : str
    camera_index : int
    filename :str
    filesize : int
    width : int
    height : int

class DefectModel(BaseModel):
    idx : int
    x1 : int
    y1 : int
    x2 : int
    y2 : int
    #filename: str
    
class InferenceResultModel(BaseModel):
    is_success : bool
    error_message : str
    inspection_result : str # good or bad
    bad_object_count : int
    raw_image_path : str
    result_image_path : str
    defects : list[DefectModel] # [[x1, y1, x2, y2], [], [], ....]
    
