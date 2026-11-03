import os
import math
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor
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

PHY_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
BGC_DATASET = "cmems_mod_glo_bgc_anfc_0.25deg_P1D-m"

executor = ThreadPoolExecutor(max_workers=4)


def _open_dataset(dataset_id, variables, lat, lng):
    """Thread içinde çalışır — event loop çakışması olmaz."""
    return copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_latitude=lat - 0.15,
        maximum_latitude=lat + 0.15,
        minimum_longitude=lng - 0.15,
        maximum_longitude=lng + 0.15,
    )


async def get_cmems_value(dataset_id, variable, lat, lng):
    try:
        loop = asyncio.get_event_loop()
        ds = await loop.run_in_executor(
            executor,
            lambda: _open_dataset(dataset_id, [variable], lat, lng)
        )
        if ds is None:
            return None
        ds.load()
        arr = ds[variable].isel(time=-1)
        if "depth" in arr.dims:
            arr = arr.isel(depth=0)
        val = float(arr.mean().values)
        return None if math.isnan(val) else round(val, 3)
    except Exception as e:
        print(f"CMEMS error {variable}: {e}")
        return None


def get_fishbase_food(spec_code):
    try:
        r = requests.get(
            "https://fishbase.ropensci.org/fooditems",
            params={"SpecCode": spec_code},
            timeout=8,
        )
        data = r.json().get("data", [])
        return list({d["FoodII"] for d in data if d.get("FoodII")})[:10]
    except:
        return []


@app.get("/marine")
async def marine(lat: float, lng: float):
    result = {"lat": lat, "lng": lng, "physics": {}, "biochemistry": {}}

    physics_vars = {
        "thetao": "sst", "so": "salinity",
        "uo": "current_east", "vo": "current_north",
        "zos": "sea_level", "mlotst": "mixed_layer_depth",
    }
    tasks = {key: get_cmems_value(PHY_DATASET, var, lat, lng)
             for var, key in physics_vars.items()}
    for key, coro in tasks.items():
        result["physics"][key] = await coro

    uo = result["physics"].get("current_east")
    vo = result["physics"].get("current_north")
    if isinstance(uo, float) and isinstance(vo, float):
        result["physics"]["current_speed"]     = round(math.sqrt(uo**2 + vo**2), 3)
        result["physics"]["current_direction"] = round((math.degrees(math.atan2(uo, vo)) + 360) % 360, 1)

    bgc_vars = {
        "chl": "chlorophyll", "phyc": "phytoplankton",
        "o2": "oxygen", "no3": "nitrate", "po4": "phosphate",
        "ph": "ph", "nppv": "primary_production",
    }
    for var, key in bgc_vars.items():
        result["biochemistry"][key] = await get_cmems_value(BGC_DATASET, var, lat, lng)

    return result


@app.get("/debug")
async def debug():
    result = {}

    # Modülde hangi fonksiyonlar var
    result["module_funcs"] = [f for f in dir(copernicusmarine) if not f.startswith("_")]

    # read_dataframe dene
    loop = asyncio.get_event_loop()
    try:
        df = await loop.run_in_executor(
            executor,
            lambda: copernicusmarine.read_dataframe(
                dataset_id=PHY_DATASET,
                variables=["thetao"],
                minimum_latitude=40.9,
                maximum_latitude=41.1,
                minimum_longitude=28.9,
                maximum_longitude=29.1,
            )
        )
        result["read_dataframe_type"] = str(type(df))
        if df is not None and not df.empty:
            result["thetao"] = float(df["thetao"].dropna().iloc[-1])
    except Exception as e:
        result["read_dataframe_error"] = str(e)[:300]

    # Login testi
    try:
        login_result = await loop.run_in_executor(
            executor,
            lambda: copernicusmarine.login(
                username=CMEMS_USER,
                password=CMEMS_PASS,
                overwrite_configuration_file=True,
            )
        )
        result["login_result"] = str(login_result)
        result["login_type"] = str(type(login_result))
    except Exception as e:
        result["login_error"] = str(e)[:300]

    # Describe dene - catalog erişimi
    try:
        cat = await loop.run_in_executor(
            executor,
            lambda: copernicusmarine.describe(
                contains=["GLOBAL_ANALYSISFORECAST_PHY"],
            )
        )
        result["describe_type"] = str(type(cat))
        if cat is not None and hasattr(cat, "products"):
            result["products_count"] = len(cat.products)
    except Exception as e:
        result["describe_error"] = str(e)[:300]

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
