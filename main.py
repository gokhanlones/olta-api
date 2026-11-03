
import os
import math
from datetime import datetime, timedelta

import copernicusmarine

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="OLTA Marine API",
    version="2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ==================================================
# COPERNICUS CREDENTIALS
# ==================================================

CMEMS_USER = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS = os.environ.get("CMEMS_PASSWORD")


# ==================================================
# DATASET IDS
# ==================================================

PHYSICS_DATASET = (
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
)

CURRENTS_DATASET = (
    "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
)

CHLOROPHYLL_DATASET = (
    "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
)


# ==================================================
# COMMON FUNCTIONS
# ==================================================

def validate_coordinates(lat, lng):

    if not (-90 <= lat <= 90):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz enlem"
        )

    if not (-180 <= lng <= 180):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz boylam"
        )


def get_dataset(
    dataset_id,
    variables,
    lat,
    lng,
    depth=False,
    start_date=None,
    end_date=None
):

    if not CMEMS_USER or not CMEMS_PASS:
        raise Exception(
            "Copernicus kullanıcı bilgileri tanımlanmamış."
        )

    params = {
        "dataset_id": dataset_id,
        "username": CMEMS_USER,
        "password": CMEMS_PASS,
        "variables": variables,
        "minimum_latitude": max(-90, lat - 0.1),
        "maximum_latitude": min(90, lat + 0.1),
        "minimum_longitude": max(-180, lng - 0.1),
        "maximum_longitude": min(180, lng + 0.1),
    }

    if depth:
        params["minimum_depth"] = 0
        params["maximum_depth"] = 5

    if start_date:
        params["start_datetime"] = start_date

    if end_date:
        params["end_datetime"] = end_date

    return copernicusmarine.open_dataset(**params)


def get_value(ds, variable, lat, lng):

    if variable not in ds:
        raise Exception(
            f"{variable} bulunamadı. "
            f"Mevcut değişkenler: {list(ds.data_vars)}"
        )

    data = ds[variable]

    if "latitude" in data.dims:
        data = data.sel(
            latitude=lat,
            method="nearest"
        )

    if "longitude" in data.dims:
        data = data.sel(
            longitude=lng,
            method="nearest"
        )

    if "time" in data.dims:
        data = data.isel(time=-1)

    if "depth" in data.dims:
        data = data.isel(depth=0)

    value = float(data.compute().values)

    if not math.isfinite(value):
        return None

    return value


def get_dataset_time(ds):

    if "time" not in ds.coords:
        return None

    try:
        return str(ds["time"].values[-1])
    except Exception:
        return None


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "user": bool(CMEMS_USER),
        "version": "2.1"
    }


# ==================================================
# DATASET DEBUG
# ==================================================

@app.get("/debug")
def debug(
    lat: float = 41.0,
    lng: float = 29.0
):

    validate_coordinates(lat, lng)

    datasets = {
        "physics": {
            "id": PHYSICS_DATASET,
            "variables": [
                "tob",
                "sob",
                "mlotst",
                "zos"
            ],
            "depth": False
        },

        "currents": {
            "id": CURRENTS_DATASET,
            "variables": [
                "uo",
                "vo"
            ],
            "depth": True
        },

        "chlorophyll": {
            "id": CHLOROPHYLL_DATASET,
            "variables": [
                "CHL"
            ],
            "depth": False
        }
    }

    results = {}

    for name, config in datasets.items():

        ds = None

        try:

            ds = get_dataset(
                config["id"],
                config["variables"],
                lat,
                lng,
                config["depth"]
            )

            results[name] = {
                "ok": True,
                "dataset_id": config["id"],
                "variables": list(ds.data_vars),
                "dimensions": dict(ds.sizes),
                "coordinates": list(ds.coords),
                "latest_time": get_dataset_time(ds)
            }

        except Exception as e:

            results[name] = {
                "ok": False,
                "dataset_id": config["id"],
                "error": str(e)[:500]
            }

        finally:

            if ds is not None:
                ds.close()

    return {
        "ok": True,
        "lat": lat,
        "lng": lng,
        "datasets": results
    }


# ==================================================
# MAIN MARINE API
# ==================================================

@app.get("/marine")
def marine(
    lat: float,
    lng: float
):

    validate_coordinates(lat, lng)

    result = {
        "ok": True,
        "lat": lat,
        "lng": lng,

        "temperature": None,
        "salinity": None,
        "mixing_layer_depth": None,
        "sea_level": None,

        "current_eastward": None,
        "current_northward": None,
        "current_speed": None,
        "current_direction": None,

        "chlorophyll": None,

        "errors": {}
    }

    # ----------------------------------------------
    # 1. PHYSICAL DATA
    # ----------------------------------------------

    ds = None

    try:

        ds = get_dataset(
            PHYSICS_DATASET,
            ["tob", "sob", "mlotst", "zos"],
            lat,
            lng
        )

        result["temperature"] = get_value(
            ds, "tob", lat, lng
        )

        result["salinity"] = get_value(
            ds, "sob", lat, lng
        )

        result["mixing_layer_depth"] = get_value(
            ds, "mlotst", lat, lng
        )

        result["sea_level"] = get_value(
            ds, "zos", lat, lng
        )

        result["physics_time"] = get_dataset_time(ds)

    except Exception as e:

        result["errors"]["physics"] = str(e)[:500]

    finally:

        if ds is not None:
            ds.close()

    # ----------------------------------------------
    # 2. OCEAN CURRENTS
    # ----------------------------------------------

    ds = None

    try:

        ds = get_dataset(
            CURRENTS_DATASET,
            ["uo", "vo"],
            lat,
            lng,
            depth=True
        )

        u = get_value(ds, "uo", lat, lng)
        v = get_value(ds, "vo", lat, lng)

        result["current_eastward"] = u
        result["current_northward"] = v

        if u is not None and v is not None:

            speed = math.sqrt(u ** 2 + v ** 2)

            result["current_speed"] = speed

            result["current_direction"] = (
                math.degrees(math.atan2(u, v)) + 360
            ) % 360

        result["currents_time"] = get_dataset_time(ds)

    except Exception as e:

        result["errors"]["currents"] = str(e)[:500]

    finally:

        if ds is not None:
            ds.close()

    # ----------------------------------------------
    # 3. CHLOROPHYLL
    # ----------------------------------------------

    ds = None

    try:

        ds = get_dataset(
            CHLOROPHYLL_DATASET,
            ["CHL"],
            lat,
            lng
        )

        result["chlorophyll"] = get_value(
            ds,
            "CHL",
            lat,
            lng
        )

        result["chlorophyll_time"] = get_dataset_time(ds)

    except Exception as e:

        result["errors"]["chlorophyll"] = str(e)[:500]

    finally:

        if ds is not None:
            ds.close()

    # ----------------------------------------------
    # STATUS
    # ----------------------------------------------

    if result["errors"]:
        result["status"] = "partial"
    else:
        result["status"] = "ok"

    return result


# ==================================================
# CHLOROPHYLL HISTORY - LAST N DAYS
# ==================================================

@app.get("/chlorophyll-history")
def chlorophyll_history(
    lat: float,
    lng: float,
    days: int = 30
):

    validate_coordinates(lat, lng)

    if not (1 <= days <= 90):
        raise HTTPException(
            status_code=400,
            detail="Gün sayısı 1-90 arasında olmalı."
        )

    ds = None

    try:

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        start_string = start_date.strftime(
            "%Y-%m-%dT00:00:00"
        )

        end_string = end_date.strftime(
            "%Y-%m-%dT23:59:59"
        )

        ds = get_dataset(
            CHLOROPHYLL_DATASET,
            ["CHL"],
            lat,
            lng,
            start_date=start_string,
            end_date=end_string
        )

        if "CHL" not in ds:
            raise Exception(
                f"CHL bulunamadı: {list(ds.data_vars)}"
            )

        data = ds["CHL"]

        if "latitude" in data.dims:
            data = data.sel(
                latitude=lat,
                method="nearest"
            )

        if "longitude" in data.dims:
            data = data.sel(
                longitude=lng,
                method="nearest"
            )

        if "time" not in data.dims:
            raise Exception(
                "CHL verisinde zaman boyutu bulunamadı."
            )

        values = data.compute().values
        times = ds["time"].values

        records = []
        valid_values = []

        for timestamp, raw_value in zip(times, values):

            value = float(raw_value)

            if math.isfinite(value):

                valid_values.append(value)

                records.append({
                    "date": str(timestamp)[:10],
                    "chlorophyll": value
                })

            else:

                records.append({
                    "date": str(timestamp)[:10],
                    "chlorophyll": None
                })

        return {
            "ok": True,
            "lat": lat,
            "lng": lng,

            "requested_days": days,

            "total_records": len(records),

            "valid_records": len(valid_values),

            "missing_records": (
                len(records) - len(valid_values)
            ),

            "average_chlorophyll": (
                sum(valid_values) / len(valid_values)
                if valid_values else None
            ),

            "maximum_chlorophyll": (
                max(valid_values)
                if valid_values else None
            ),

            "minimum_chlorophyll": (
                min(valid_values)
                if valid_values else None
            ),

            "records": records
        }

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=str(e)[:1000]
        )

    finally:

        if ds is not None:
            ds.close()
