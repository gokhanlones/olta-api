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

# Copernicusmarine 2.x doğru env var isimleri
CMEMS_USER = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS = os.environ.get("CMEMS_PASSWORD")

if CMEMS_USER:
    os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = CMEMS_USER
if CMEMS_PASS:
    os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = CMEMS_PASS


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
        arr = ds[variable].isel(time=-1)
        if "depth" in arr.dims:
            arr = arr.isel(depth=0)
        val = float(arr.mean().values)
        if math.isnan(val):
            return None
        return round(val, 3)
    except Exception as e:
        print(f"CMEMS error {dataset_id}/{variable}: {e}")
        return None


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
    result = {
        "lat": lat,
        "lng": lng,
        "physics": {},
        "biochemistry": {},
    }

    phys_dataset = "cmems_mod_glo_phy-all_anfc_0.083deg_PT6H-i"
    physics_vars = {
        "thetao": "sst",
        "so":     "salinity",
        "uo":     "current_east",
        "vo":     "current_north",
        "zos":    "sea_level",
        "mlotst": "mixed_layer_depth",
    }
    for var, key in physics_vars.items():
        result["physics"][key] = get_cmems_value(phys_dataset, var, lat, lng)

    uo = result["physics"].get("current_east")
    vo = result["physics"].get("current_north")
    if uo is not None and vo is not None:
        result["physics"]["current_speed"]     = round(math.sqrt(uo**2 + vo**2), 3)
        result["physics"]["current_direction"] = round((math.degrees(math.atan2(uo, vo)) + 360) % 360, 1)

    bgc_dataset = "cmems_mod_glo_bgc-all_anfc_0.25deg_P1D-m"
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
        result["biochemistry"][key] = get_cmems_value(bgc_dataset, var, lat, lng)

    return result


@app.get("/foodchain")
def foodchain(spec_code: int):
    return {"prey": get_fishbase_food(spec_code)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cmems_user": bool(CMEMS_USER),
        "env_set": bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")),
    }
