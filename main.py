import os
import math
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import copernicusmarine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

CMEMS_USER = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS = os.environ.get("CMEMS_PASSWORD")

if CMEMS_USER:
    os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = CMEMS_USER
if CMEMS_PASS:
    os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = CMEMS_PASS


@app.get("/health")
def health():
    return {"status": "ok", "user": bool(CMEMS_USER)}


@app.get("/debug")
def debug():
    result = {}

    # Login
    try:
        r = copernicusmarine.login(
            username=CMEMS_USER,
            password=CMEMS_PASS,
            overwrite_configuration_file=True,
        )
        result["login"] = str(r)
    except Exception as e:
        result["login_error"] = str(e)[:200]

    # open_dataset - credentials direkt parametre
    try:
        import inspect
        sig = inspect.signature(copernicusmarine.open_dataset)
        result["open_dataset_sig"] = str(sig)[:300]
    except Exception as e:
        result["sig_error"] = str(e)[:100]

    try:
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            username=CMEMS_USER,
            password=CMEMS_PASS,
        )
        result["ds_type"] = str(type(ds))
        result["ds_none"] = ds is None
        if ds is not None:
            result["vars"] = list(ds.data_vars)
            result["dims"] = dict(ds.sizes)
    except Exception as e:
        result["open_error"] = str(e)[:300]

    return result


@app.get("/marine")
def marine(lat: float, lng: float):
    return {"lat": lat, "lng": lng, "status": "coming soon"}
