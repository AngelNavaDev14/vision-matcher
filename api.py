import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from matcher import TemplateMatcher

app = FastAPI(title="Vision Matcher API")
matcher = TemplateMatcher(templates_dir="templates", threshold=0.80)

@app.post("/detect")
async def detect(
    screenshot: UploadFile = File(...),
    label: str = "kickoff"
):
    data = await screenshot.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Imagen inválida")

    result = matcher.match(img, label)

    if result is None:
        return JSONResponse({"found": False, "match": None})

    return JSONResponse({
        "found": result.found,
        "match": {
            "x": result.x,
            "y": result.y,
            "w": result.w,
            "h": result.h,
            "confidence": result.confidence,
            "label": result.label
        }
    })

@app.post("/detect_all")
async def detect_all(screenshot: UploadFile = File(...)):
    data = await screenshot.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Imagen inválida")

    results = matcher.match_any(img)
    return JSONResponse({
        "found": len(results) > 0,
        "matches": [
            {"x": r.x, "y": r.y, "w": r.w, "h": r.h,
             "confidence": r.confidence, "label": r.label}
            for r in results
        ]
    })

@app.get("/health")
def health():
    return {"status": "ok", "templates_loaded": list(matcher.templates.keys())}
