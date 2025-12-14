import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from kaki_analysis import analyze_kaki_image

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = analyze_kaki_image(file_bytes)
    return result

if __name__ == "__main__":
    # Renderの環境変数 PORT を取得。なければ 8000
    port = int(os.environ.get("PORT", 8000))
    # hostは必ず "0.0.0.0" に設定する
    uvicorn.run(app, host="0.0.0.0", port=port)