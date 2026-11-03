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

# Doğru dataset ID'leri
PHY_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
BGC_DATASET = "cmems_mod_glo_bgc_anfc_0.25deg_P1D-m"

PHYSICS_DATASETS = {v: PHY_DATASET for v in ["thetao","so","uo","vo","zos","mlotst"]}
BGC_DATASETS     = {v: BGC_DATASET for v in ["chl","phyc","o2","no3","po4","ph","nppv"]}


def get_cmems_value(dataset_id, variable, lat, lng):
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            variables=[variable],
            minimum_latitude=lat - 0.15,
            maximum_latitude=lat + 0.15,
            minimum_longitude=lng - 0.15,
            maximum_longitude=lng + 0.15,
        )
        if ds is None:
            return f"ERR: ds is None for {dataset_id}"
        if variable not in ds:
            return f"ERR: {variable} not in ds, keys={list(ds.keys())[:5]}"
        arr = ds[variable].isel(time=-1)
        if "depth" in arr.dims:
            arr = arr.isel(depth=0)
        val = float(arr.mean().values)
        if math.isnan(val):
            return None
        return round(val, 3)
    except Exception as e:
        print(f"CMEMS error {dataset_id}/{variable}: {e}")
        return f"ERR: {str(e)[:120]}"


def get_fishbase_food(spec_code):
    try:
        r = requests.get(
            "https://fishbase.ropensci.org/fooditems",
            params={"SpecCode": spec_code},
            timeout=8,
        )
        data = r.json().get("data", [])
        prey = list({d["FoodII"] for d in data if d.get("FoodII")})
        return prey[:10]
    except:
        return []


@app.get("/marine")
def marine(lat: float, lng: float):
    result = {"lat": lat, "lng": lng, "physics": {}, "biochemistry": {}}

    physics_vars = {
        "thetao": "sst",
        "so":     "salinity",
        "uo":     "current_east",
        "vo":     "current_north",
        "zos":    "sea_level",
        "mlotst": "mixed_layer_depth",
    }
    for var, key in physics_vars.items():
        result["physics"][key] = get_cmems_value(PHYSICS_DATASETS[var], var, lat, lng)

    uo = result["physics"].get("current_east")
    vo = result["physics"].get("current_north")
    if isinstance(uo, float) and isinstance(vo, float):
        result["physics"]["current_speed"]     = round(math.sqrt(uo**2 + vo**2), 3)
        result["physics"]["current_direction"] = round((math.degrees(math.atan2(uo, vo)) + 360) % 360, 1)

    bgc_vars = {
        "chl":  "chlorophyll",
        "phyc": "phytoplankton",
        "o2":   "oxygen",
        "no3":  "nitrate",
        "po4":  "phosphate",
        "ph":   "ph",
        "nppv": "primary_production",
    }
    for var, key in bgc_vars.items():
        result["biochemistry"][key] = get_cmems_value(BGC_DATASETS[var], var, lat, lng)

    return result


@app.get("/foodchain")
def foodchain(spec_code: int):
    return {"prey": get_fishbase_food(spec_code)}


@app.get("/debug")
def debug():
    result = {}

    # Test 1: sadece dataset_id ile dene
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        )
        result["type"] = str(type(ds))
        result["is_none"] = ds is None
        if ds is not None:
            result["vars"] = list(ds.data_vars)[:10]
    except Exception as e:
        result["test1_error"] = str(e)[:300]

    # Test 2: coğrafi filtre + load()
    try:
        ds2 = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            minimum_latitude=39.9,
            maximum_latitude=40.1,
            minimum_longitude=-30.1,
            maximum_longitude=-29.9,
        )
        result["test2_type"] = str(type(ds2))
        if ds2 is not None:
            ds2.load()
            val = float(ds2["thetao"].isel(time=-1).mean().values)
            result["thetao"] = round(val, 3)
    except Exception as e:
        result["test2_error"] = str(e)[:300]

    return result

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cmems_user": bool(CMEMS_USER),
        "env_set": bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")),
    }
