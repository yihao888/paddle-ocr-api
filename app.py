from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import logging

# 关闭 PaddleOCR 日志（避免刷屏）
logging.getLogger("ppocr").setLevel(logging.WARNING)

# 初始化 OCR 引擎（只初始化一次）
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='ch',          # 支持中英文，如只需英文可改为 'en'
    show_log=False,
    use_gpu=False       # 强制使用 CPU（Hugging Face Spaces 无 GPU）
)

app = FastAPI(
    title="PaddleOCR API",
    description="通过图片 URL 进行 OCR 识别"
)

class ImageUrlRequest(BaseModel):
    image_url: str

@app.post("/ocr")
async def ocr_endpoint(request: ImageUrlRequest):
    try:
        # 1. 下载图片
        response = requests.get(request.image_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法下载图片，请检查 URL 是否有效")
        
        # 2. 打开图片并转为 RGB
        image = Image.open(BytesIO(response.content)).convert('RGB')
        img_array = np.array(image)

        # 3. 执行 OCR
        result = ocr_engine.ocr(img_array, cls=True)

        # 4. 提取文本
        texts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]  # 提取识别出的文字
                confidence = line[1][1]  # 置信度（可选）
                texts.append(text)

        return {
            "success": True,
            "texts": texts,
            "count": len(texts)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
