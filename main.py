import os
import math
import copernicusmarine

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

DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"


def get_dataset(lat, lng, variables):
    return copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        username=CMEMS_USER,
        password=CMEMS_PASS,
        variables=variables,
        minimum_latitude=lat - 0.05,
        maximum_latitude=lat + 0.05,
        minimum_longitude=lng - 0.05,
        maximum_longitude=lng + 0.05,
    )


def get_value(ds, variable, lat, lng):
    if variable not in ds:
        return None

    point = ds[variable].sel(
        latitude=lat,
        longitude=lng,
        method="nearest",
    )

    if "time" in point.dims:
        point = point.isel(time=-1)

    value = float(point.compute().values)

    return value if math.isfinite(value) else None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "user": bool(CMEMS_USER),
    }


@app.get("/debug")
def debug():
    try:
        ds = get_dataset(
            41.0,
            29.0,
            ["tob", "sob", "mlotst", "zos"],
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
        ds = get_dataset(
            lat,
            lng,
            ["tob", "sob", "mlotst", "zos"],
        )

        return {
            "ok": True,
            "lat": lat,
            "lng": lng,
            "temperature": get_value(ds, "tob", lat, lng),
            "salinity": get_value(ds, "sob", lat, lng),
            "mixing_layer_depth": get_value(
                ds, "mlotst", lat, lng
            ),
            "sea_level": get_value(ds, "zos", lat, lng),
            "status": "ok",
        }

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        )
