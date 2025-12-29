## Using this command to fix path error if manual run api in container:
``` bash
python -c "import sys; sys.path.append('./')"
```
## then run:
``` bash
uvicorn src.ahs.api.fast_api:app --host 0.0.0.0 --port 8000
```