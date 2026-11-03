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

# Her değişken için doğru dataset ID
PHYSICS_DATASETS = {
    "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i",
    "so":     "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i",
    "uo":     "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
    "vo":     "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
    "zos":    "cmems_mod_glo_phy-ssh_anfc_0.083deg_PT6H-i",
    "mlotst": "cmems_mod_glo_phy-mnhcl_anfc_0.083deg_PT6H-i",
}

BGC_DATASETS = {
    "chl":  "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
    "phyc": "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
    "o2":   "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
    "no3":  "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    "po4":  "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    "ph":   "cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
    "nppv": "cmems_mod_glo_bgc-pp_anfc_0.25deg_P1D-m",
}


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cmems_user": bool(CMEMS_USER),
        "env_set": bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")),
    }
