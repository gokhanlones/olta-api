import os
import math
import copernicusmarine
import xarray as xr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

CMEMS_USER = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS = os.environ.get("CMEMS_PASSWORD")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "user": bool(CMEMS_USER),
    }


@app.get("/debug")
def debug():
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            username=CMEMS_USER,
            password=CMEMS_PASS,
            minimum_latitude=40.9,
            maximum_latitude=41.1,
            minimum_longitude=28.9,
            maximum_longitude=29.1,
        )

        return {
            "ok": True,
            "variables": list(ds.data_vars),
            "dimensions": dict(ds.sizes),
        }

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        )


@app.get("/marine")
def marine(lat: float, lng: float):
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz koordinat",
        )

    try:
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            username=CMEMS_USER,
            password=CMEMS_PASS,
            variables=["tob"],
            minimum_latitude=lat - 0.05,
            maximum_latitude=lat + 0.05,
            minimum_longitude=lng - 0.05,
            maximum_longitude=lng + 0.05,
        )

        if ds is None:
            raise HTTPException(
                status_code=502,
                detail="Copernicus veri döndürmedi",
            )

        if "tob" not in ds:
            raise HTTPException(
                status_code=502,
                detail=f"tob bulunamadı: {list(ds.data_vars)}",
            )

        point = ds["tob"].sel(
            latitude=lat,
            longitude=lng,
            method="nearest",
        ).isel(time=-1)

        value = float(point.compute().values)

        if not math.isfinite(value):
            return {
                "ok": True,
                "temperature": None,
                "status": "no_data",
            }

        return {
            "ok": True,
            "lat": lat,
            "lng": lng,
            "temperature": value,
            "unit": "°C",
            "variable": "tob",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        )
