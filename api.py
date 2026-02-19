from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from data_processor import preprocess_training_ready

app = FastAPI()

@app.post('/preprocess')
async def preprocess(file: UploadFile = File(...)):
    # save uploaded file to a temp location
    try:
        import traceback
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.csv'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        result = preprocess_training_ready(tmp.name)
        response = {
            'report': result.get('report', {}),
            'train_shape': (list(result['train'][0].shape), list(result['train'][1].shape)),
            'val_shape': (list(result['val'][0].shape), list(result['val'][1].shape)),
            'test_shape': (list(result['test'][0].shape), list(result['test'][1].shape)),
        }
        return JSONResponse(content=response)
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={'error': str(e), 'trace': traceback.format_exc()})
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
