import base64, io
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve
from fastapi import FastAPI
import easyocr
from PIL import Image
from pydantic import BaseModel

config = Config()
config.bind = ["localhost:8000"]

class ImageOCR(BaseModel):
    base64_string: str
    language: str

app = FastAPI()

@app.post("/ocr")
async def ocr(image: ImageOCR):
    reader = easyocr.Reader([image.language], gpu=False)
    image_data = base64.b64decode(str(image.base64_string))
    image_to_extract = Image.open(io.BytesIO(image_data))
    result = reader.readtext(image_to_extract)
    json_result = list(
        map(
            lambda item: {
                "boundaryBox": [
                    [item[0][0][0].item(), item[0][0][1].item()],
                    [item[0][1][0].item(), item[0][1][1].item()],
                    [item[0][2][0].item(), item[0][2][1].item()],
                    [item[0][3][0].item(), item[0][3][1].item()],
                ],
                "text": item[1],
                "confidence": item[2].item(),
            },
            result,
        )
    )
    return json_result

if __name__ == "__main__":
    asyncio.run(serve(app, config))
