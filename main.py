import os
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


def get_cmems_value(dataset_id, variable, lat, lng):
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            variables=[variable],
            minimum_latitude=lat - 0.1,
            maximum_latitude=lat + 0.1,
            minimum_longitude=lng - 0.1,
            maximum_longitude=lng + 0.1,
            username=CMEMS_USER,
            password=CMEMS_PASS,
        )
        val = float(ds[variable].isel(time=-1).mean().values)
        return round(val, 3)
    except Exception as e:
        return None


def get_fishbase_food(spec_code):
    try:
        r = requests.get(
            "https://fishbase.ropensci.org/fooditems",
            params={"SpecCode": spec_code},
            timeout=8,
        )
        data = r.json()["data"]
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

    # ── Fizik — Global Ocean Physics (GLOBAL_ANALYSISFORECAST_PHY_001_024)
    physics_vars = {
        "thetao": "sst",
        "so":     "salinity",
        "uo":     "current_east",
        "vo":     "current_north",
        "zos":    "sea_level",
        "mlotst": "mixed_layer_depth",
    }
    for var, key in physics_vars.items():
        result["physics"][key] = get_cmems_value(
            "cmems_mod_glo_phy-all_anfc_0.083deg_PT6H-i", var, lat, lng
        )

    # Akıntı hız ve yön hesapla
    uo = result["physics"].get("current_east")
    vo = result["physics"].get("current_north")
    if uo is not None and vo is not None:
        import math
        result["physics"]["current_speed"] = round(math.sqrt(uo**2 + vo**2), 3)
        result["physics"]["current_direction"] = round(
            (math.degrees(math.atan2(uo, vo)) + 360) % 360, 1
        )

    # ── Biyokimya — Global BGC (GLOBAL_ANALYSISFORECAST_BGC_001_028)
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
        result["biochemistry"][key] = get_cmems_value(
            "cmems_mod_glo_bgc-all_anfc_0.25deg_P1D-m", var, lat, lng
        )

    return result


@app.get("/foodchain")
def foodchain(spec_code: int):
    return {"prey": get_fishbase_food(spec_code)}


@app.get("/health")
def health():
    return {"status": "ok"}
