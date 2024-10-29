import base64, io, json
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve
from fastapi import FastAPI
import easyocr
from PIL import Image
from pydantic import BaseModel
import pymrio
from upstash_redis import Redis
from transformers import AutoModelForSequenceClassification
import torch

config = Config()
config.bind = ["localhost:8000"]


class ImageOCR(BaseModel):
    image: str
    language: str


app = FastAPI()

model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # or 'cpu' if no GPU is available
model.eval()


class RankingBody(BaseModel):
    query: str
    documents: list[str]


@app.post("/ocr")
async def ocr(image: ImageOCR):
    reader = easyocr.Reader([image.language], gpu=False)
    image_data = base64.b64decode(str(image.image))
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

@app.get("/update-google-taxonomy-emission-factors")
async def update_google_taxonomy_emission_factors():
    for physicalGoodsCategory in []:
        redis_key = "PRODUCT:" + str(physicalGoodsCategory["id"])
        #result = redis.hset(redis_key, values=physicalGoodsCategory)
        print(redis_key)
        print(result)
    print("finished")
    print("really finished")

@app.get("/update-db-exio3-year-region")
async def update_db_exio3_year_region():
    years = ["2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011", "2010"]
    for year in years:
        print("year" + year + "calculating...")
        exio3_path = "IOT_" + year + "_ixi.zip"
        exio3 = pymrio.parse_exiobase3(exio3_path)
        exio3.calc_all()
        impact = "GHG emissions AR5 (GWP100) | GWP100 (IPCC, 2010)"
        regions = list(exio3.get_regions())
        sectors = list(exio3.get_sectors())
        impact_key = "CO2E:" + year + "-EXIO3-I"
        for index, region in enumerate(regions):
            redis_object = {}
            redis_object["region"] = region
            redis_object["version"] = "3.8.2"
            redis_object["impact"] = impact
            for index, sector in enumerate(sectors):
                redis_object[index] = str(exio3.impacts.M.loc[impact][region].values[index])
            impact_key_with_index = impact_key + ":" + region
            # result = redis.hset(impact_key_with_index, values=redis_object)
            print(impact_key_with_index)
            print(result)
        print("finished" + str(year))
    print("finished completely")
    print("really finished")

@app.get("/update-db-exio3-year-industry")
async def update_db_exio3_year_industry(year: int):
    exio3_path = "IOT_" + str(year) + "_ixi.zip"
    exio3 = pymrio.parse_exiobase3(exio3_path)
    exio3.calc_all()
    impact = "GHG emissions AR5 (GWP100) | GWP100 (IPCC, 2010)"
    regions = list(exio3.get_regions())
    sectors = list(exio3.get_sectors())
    impact_key = "CO2E:" + str(year) + "-EXIO3-I"
    for index, sector in enumerate(sectors):
        redis_object = {}
        redis_object["sector"] = sector
        redis_object["version"] = "3.8.2"
        redis_object["impact"] = impact
        for region in regions:
            redis_object[region] = str(exio3.impacts.M.loc[impact][region].values[index])
        impact_key_with_index = impact_key + ":" + str(index)
        # result = redis.hset(impact_key_with_index, values=redis_object)
        print(impact_key_with_index)
        print(result)
    print("finished" + str(year))


@app.post("/rankingv2")
async def ranking(ranking_body: RankingBody):
    query = ranking_body.query
    documents = ranking_body.documents
    sentence_pairs = [[query, doc] for doc in documents]
    scores = model.compute_score(sentence_pairs, max_length=1024)
    return scores


if __name__ == "__main__":
    asyncio.run(serve(app, config), debug=True)
