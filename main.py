import os
import json
import math
import time
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import copernicusmarine
import firebase_admin
from firebase_admin import credentials, firestore as fs

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(title="OLTA Marine API", version="3.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ==================================================
# FIREBASE ADMIN
# ==================================================

_db = None

_fb_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
if _fb_key and not firebase_admin._apps:
    try:
        _cred = credentials.Certificate(json.loads(_fb_key))
        firebase_admin.initialize_app(_cred)
        _db = fs.client()
    except Exception as _e:
        print(f"Firebase init error: {_e}")

CRON_SECRET = os.environ.get("CRON_SECRET", "olta2024")

# ==================================================
# COPERNICUS CREDENTIALS
# ==================================================

CMEMS_USER = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS = os.environ.get("CMEMS_PASSWORD")

# ==================================================
# DATASET IDS
# ==================================================

PHYSICS_DATASET  = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
CURRENTS_DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"

CHL_BLACKSEA_NRT     = "cmems_obs-oc_blk_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D"
CHL_BLACKSEA_HISTORY = "cmems_obs-oc_blk_bgc-plankton_my_l4-gapfree-multi-1km_P1D"
CHL_GLOBAL           = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"

BGC_BIO_DATASET      = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
BGC_NUT_DATASET      = "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m"
BGC_PFT_DATASET      = "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m"
BGC_PLANKTON_DATASET = "cmems_mod_glo_bgc-plankton_anfc_0.25deg_P1D-m"

CHL_SOURCES = [
    {"name": "blacksea_nrt_1km",     "id": CHL_BLACKSEA_NRT,     "regional": True},
    {"name": "blacksea_history_1km", "id": CHL_BLACKSEA_HISTORY,  "regional": True},
    {"name": "global_4km",           "id": CHL_GLOBAL,            "regional": False},
]

# ==================================================
# COMMON FUNCTIONS
# ==================================================

def validate_coordinates(lat, lng):
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Geçersiz enlem")
    if not (-180 <= lng <= 180):
        raise HTTPException(status_code=400, detail="Geçersiz boylam")


def get_dataset(dataset_id, variables, lat, lng, depth=False,
                start_date=None, end_date=None, buffer=0.1):
    if not CMEMS_USER or not CMEMS_PASS:
        raise Exception("Copernicus kullanıcı bilgileri tanımlanmamış.")
    params = {
        "dataset_id": dataset_id,
        "username": CMEMS_USER,
        "password": CMEMS_PASS,
        "variables": variables,
        "minimum_latitude":  max(-90,  lat - buffer),
        "maximum_latitude":  min(90,   lat + buffer),
        "minimum_longitude": max(-180, lng - buffer),
        "maximum_longitude": min(180,  lng + buffer),
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
        raise Exception(f"{variable} bulunamadı: {list(ds.data_vars)}")
    data = ds[variable]
    lat_name = next((n for n in ["latitude","lat"] if n in data.dims), None)
    lng_name = next((n for n in ["longitude","lon"] if n in data.dims), None)
    if lat_name: data = data.sel({lat_name: lat}, method="nearest")
    if lng_name: data = data.sel({lng_name: lng}, method="nearest")
    if "time"  in data.dims: data = data.isel(time=-1)
    if "depth" in data.dims: data = data.isel(depth=0)
    value = float(np.asarray(data.compute().values).squeeze())
    return None if not math.isfinite(value) else value


def get_dataset_time(ds):
    if "time" not in ds.coords:
        return None
    try:
        return str(ds["time"].values[-1])
    except Exception:
        return None


def calculate_data_age_days(timestamp):
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - parsed).total_seconds() / 86400, 2)
    except Exception:
        return None


def get_freshness_status(age_days):
    if age_days is None: return "unknown"
    if age_days < 0:     return "forecast"
    if age_days <= 3:    return "fresh"
    if age_days <= 7:    return "recent"
    if age_days <= 30:   return "stale"
    return "very_stale"

# ==================================================
# BIOGEOCHEMISTRY HELPERS
# ==================================================

def get_bgc_value(ds, variable, lat, lng):
    if variable not in ds:
        raise Exception(f"{variable} bulunamadı: {list(ds.data_vars)}")
    data = ds[variable]
    lat_name = next((n for n in ["latitude","lat"] if n in data.dims), None)
    lng_name = next((n for n in ["longitude","lon"] if n in data.dims), None)
    if lat_name: data = data.sel({lat_name: lat}, method="nearest")
    if lng_name: data = data.sel({lng_name: lng}, method="nearest")
    selected_time = None
    if "time" in data.dims:
        target_time = datetime.now(timezone.utc).replace(tzinfo=None)
        data = data.sel(time=target_time, method="nearest")
        try:
            selected_time = str(data["time"].values)
        except Exception:
            pass
    if "depth" in data.dims:
        data = data.isel(depth=0)
    value = float(np.asarray(data.compute().values).squeeze())
    return (None if not math.isfinite(value) else value), selected_time


def get_bgc_surface_depth(ds):
    if "depth" not in ds.coords:
        return None
    try:
        return float(ds["depth"].values[0])
    except Exception:
        return None

# ==================================================
# CHLOROPHYLL HELPERS
# ==================================================

def is_blacksea_region(lat, lng):
    return 40.0 <= lat <= 48.0 and 26.5 <= lng <= 42.0


def get_chlorophyll_records(source, lat, lng, start_date, end_date):
    ds = None
    try:
        ds = get_dataset(source["id"], ["CHL"], lat, lng,
                         start_date=start_date, end_date=end_date)
        if "CHL" not in ds:
            raise Exception(f"CHL bulunamadı: {list(ds.data_vars)}")
        data = ds["CHL"]
        lat_name = next((n for n in ["latitude","lat"] if n in data.dims), None)
        lng_name = next((n for n in ["longitude","lon"] if n in data.dims), None)
        if not lat_name or not lng_name:
            raise Exception(f"Koordinatlar bulunamadı: {list(data.dims)}")
        data = data.sel({lat_name: lat, lng_name: lng}, method="nearest")
        if "time" not in data.dims:
            raise Exception("CHL verisinde zaman boyutu bulunamadı.")
        values = np.asarray(data.compute().values).reshape(-1)
        times  = ds["time"].values
        records = {}
        for timestamp, raw_value in zip(times, values):
            date  = str(timestamp)[:10]
            value = float(raw_value)
            records[date] = value if (math.isfinite(value) and 0 < value <= 100) else None
        return records, None
    except Exception as e:
        return {}, str(e)[:500]
    finally:
        if ds is not None:
            ds.close()


def get_chlorophyll_history_data(lat, lng, days):
    now        = datetime.now(timezone.utc).replace(tzinfo=None)
    end_date   = now.strftime("%Y-%m-%dT23:59:59")
    start_date = (now - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")
    merged = {}
    source_status = {}
    sources = [s for s in CHL_SOURCES
               if not (s["regional"] and not is_blacksea_region(lat, lng))]
    for source in sources:
        records, error = get_chlorophyll_records(source, lat, lng, start_date, end_date)
        valid_count = sum(1 for v in records.values() if v is not None)
        source_status[source["name"]] = {
            "ok": error is None, "valid_records": valid_count,
            "total_records": len(records), "error": error
        }
        for date, value in records.items():
            if value is not None:
                if date not in merged or merged[date] is None:
                    merged[date] = {"value": value, "source": source["name"]}
            elif date not in merged:
                merged[date] = None
    return merged, source_status

# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "user":     bool(CMEMS_USER),
        "firebase": _db is not None,
        "version":  "3.5"
    }

# ==================================================
# MARINE API
# ==================================================

@app.get("/marine")
def marine(lat: float, lng: float):
    validate_coordinates(lat, lng)
    result = {
        "ok": True, "lat": lat, "lng": lng,
        "temperature": None, "salinity": None,
        "mixing_layer_depth": None, "sea_level": None,
        "current_eastward": None, "current_northward": None,
        "current_speed": None, "current_direction": None,
        "chlorophyll": None, "chlorophyll_time": None,
        "chlorophyll_source": None, "errors": {}
    }
    ds = None
    try:
        ds = get_dataset(PHYSICS_DATASET, ["tob","sob","mlotst","zos"], lat, lng)
        result["temperature"]        = get_value(ds, "tob",    lat, lng)
        result["salinity"]           = get_value(ds, "sob",    lat, lng)
        result["mixing_layer_depth"] = get_value(ds, "mlotst", lat, lng)
        result["sea_level"]          = get_value(ds, "zos",    lat, lng)
        result["physics_time"]       = get_dataset_time(ds)
    except Exception as e:
        result["errors"]["physics"] = str(e)[:500]
    finally:
        if ds: ds.close()
    ds = None
    try:
        ds = get_dataset(CURRENTS_DATASET, ["uo","vo"], lat, lng, depth=True)
        u = get_value(ds, "uo", lat, lng)
        v = get_value(ds, "vo", lat, lng)
        result["current_eastward"]  = u
        result["current_northward"] = v
        if u is not None and v is not None:
            result["current_speed"]     = math.sqrt(u**2 + v**2)
            result["current_direction"] = (math.degrees(math.atan2(u,v)) + 360) % 360
        result["currents_time"] = get_dataset_time(ds)
    except Exception as e:
        result["errors"]["currents"] = str(e)[:500]
    finally:
        if ds: ds.close()
    try:
        history, source_status = get_chlorophyll_history_data(lat, lng, 30)
        valid_dates = [(d,i) for d,i in history.items() if i is not None]
        result["chlorophyll_sources"] = source_status
        if valid_dates:
            latest_date, latest_item = max(valid_dates, key=lambda x: x[0])
            result["chlorophyll"]        = latest_item["value"]
            result["chlorophyll_time"]   = latest_date
            result["chlorophyll_source"] = latest_item["source"]
        else:
            result["errors"]["chlorophyll"] = "Geçerli klorofil ölçümü bulunamadı."
    except Exception as e:
        result["errors"]["chlorophyll"] = str(e)[:500]
    result["status"] = "partial" if result["errors"] else "ok"
    return result

# ==================================================
# HABITAT GRID CONFIG
# ==================================================

HABITAT_GRID_STEP = 0.25
HABITAT_REGION    = {"min_lat":35.0,"max_lat":42.5,"min_lng":25.0,"max_lng":42.0}
HABITAT_CACHE_TTL = 3 * 60 * 60
_habitat_cache      = {}
_habitat_cache_lock = threading.Lock()


def get_habitat_cache(key):
    with _habitat_cache_lock:
        item = _habitat_cache.get(key)
        if not item: return None
        if time.time() - item["created"] > HABITAT_CACHE_TTL:
            del _habitat_cache[key]; return None
        return item["data"]


def set_habitat_cache(key, data):
    with _habitat_cache_lock:
        _habitat_cache[key] = {"created": time.time(), "data": data}


def get_coordinate_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def get_regional_dataset(dataset_id, variables, depth=False):
    if not CMEMS_USER or not CMEMS_PASS:
        raise Exception("Copernicus kullanıcı bilgileri tanımlanmamış.")
    params = {
        "dataset_id": dataset_id,
        "username": CMEMS_USER, "password": CMEMS_PASS,
        "variables": variables,
        "minimum_latitude":  HABITAT_REGION["min_lat"],
        "maximum_latitude":  HABITAT_REGION["max_lat"],
        "minimum_longitude": HABITAT_REGION["min_lng"],
        "maximum_longitude": HABITAT_REGION["max_lng"],
    }
    if depth:
        params["minimum_depth"] = 0
        params["maximum_depth"] = 5
    return copernicusmarine.open_dataset(**params)


def regional_data_to_grid(ds, variable, target_latitudes, target_longitudes):
    if variable not in ds:
        raise Exception(f"{variable} değişkeni bulunamadı.")
    lat_name = get_coordinate_name(ds, ["latitude","lat"])
    lng_name = get_coordinate_name(ds, ["longitude","lon"])
    if not lat_name or not lng_name:
        raise Exception("Enlem/boylam koordinatları bulunamadı.")
    data = ds[variable]
    if "time"  in data.dims: data = data.sortby("time").isel(time=-1)
    if "depth" in data.dims: data = data.sortby("depth").isel(depth=0)
    data = data.squeeze(drop=True).sortby(lat_name).sortby(lng_name)
    interpolated = data.interp(
        {lat_name: target_latitudes, lng_name: target_longitudes},
        method="nearest"
    )
    values = np.asarray(interpolated.compute().values, dtype=float)
    values[~np.isfinite(values)] = np.nan
    if values.shape != (len(target_latitudes), len(target_longitudes)):
        raise Exception(f"{variable} grid boyutu beklenenden farklı.")
    return values


def bluefish_temperature_score(temperature):
    if temperature is None or not math.isfinite(temperature): return None
    minimum, maximum, center = 8.4, 27.5, 21.1
    if temperature < minimum or temperature > maximum: return 0.0
    if temperature <= center:
        score = (temperature - minimum) / (center - minimum) * 100
    else:
        score = (maximum - temperature) / (maximum - center) * 100
    return round(max(0.0, min(100.0, score)), 1)


def get_habitat_level(score):
    if score is None: return "insufficient_data"
    if score >= 75:   return "high"
    if score >= 50:   return "moderate"
    if score >= 25:   return "low"
    return "very_low"


def safe_grid_value(array, i, j):
    if array is None: return None
    value = float(array[i, j])
    return None if not math.isfinite(value) else value


def build_habitat_response(species, region):
    latitudes  = np.round(np.arange(HABITAT_REGION["min_lat"],
                                    HABITAT_REGION["max_lat"] + 0.001,
                                    HABITAT_GRID_STEP), 4)
    longitudes = np.round(np.arange(HABITAT_REGION["min_lng"],
                                    HABITAT_REGION["max_lng"] + 0.001,
                                    HABITAT_GRID_STEP), 4)
    arrays         = {"temperature":None,"salinity":None,"oxygen":None,"chlorophyll":None}
    source_times   = {}
    dataset_status = {}
    errors         = {}

    for ds_id, vars_, depth_, key_temp, key_sal_or_var in [
        (PHYSICS_DATASET, ["tob","sob"], False, "tob", "sob"),
        (BGC_BIO_DATASET, ["o2"],        True,  "o2",  None),
        (BGC_PFT_DATASET, ["chl"],       True,  "chl", None),
    ]:
        ds = None
        try:
            ds = get_regional_dataset(ds_id, vars_, depth=depth_)
            k_map = {"tob":"temperature","o2":"oxygen","chl":"chlorophyll"}
            arrays[k_map[key_temp]] = regional_data_to_grid(ds, key_temp, latitudes, longitudes)
            if key_sal_or_var:
                arrays["salinity"] = regional_data_to_grid(ds, key_sal_or_var, latitudes, longitudes)
            t_map = {"tob":"physics","o2":"oxygen","chl":"chlorophyll"}
            source_times[t_map[key_temp]]   = get_dataset_time(ds)
            dataset_status[t_map[key_temp]] = {"ok":True,"dataset_id":ds_id}
        except Exception as e:
            k = {"tob":"physics","o2":"oxygen","chl":"chlorophyll"}[key_temp]
            errors[k]         = str(e)[:500]
            dataset_status[k] = {"ok":False,"error":str(e)[:500]}
        finally:
            if ds: ds.close()

    cells        = []
    total_cells  = 0
    scored_cells = 0
    half         = HABITAT_GRID_STEP / 2

    for i, lat in enumerate(latitudes):
        for j, lng in enumerate(longitudes):
            temperature = safe_grid_value(arrays["temperature"], i, j)
            salinity    = safe_grid_value(arrays["salinity"],    i, j)
            oxygen      = safe_grid_value(arrays["oxygen"],      i, j)
            chlorophyll = safe_grid_value(arrays["chlorophyll"], i, j)
            if temperature is None and salinity is None:
                continue
            total_cells += 1
            score        = bluefish_temperature_score(temperature)
            completeness = round(sum(v is not None for v in (temperature,salinity,oxygen)) / 3 * 100)
            if score is not None: scored_cells += 1
            cells.append({
                "lat": round(float(lat),4), "lng": round(float(lng),4),
                "bounds": {
                    "south": round(float(lat-half),4), "west": round(float(lng-half),4),
                    "north": round(float(lat+half),4), "east": round(float(lng+half),4),
                },
                "temperature": round(temperature,3) if temperature is not None else None,
                "salinity":    round(salinity,3)    if salinity    is not None else None,
                "oxygen":      round(oxygen,3)      if oxygen      is not None else None,
                "oxygen_mg_l": round(oxygen*0.032,3)if oxygen      is not None else None,
                "chlorophyll": round(chlorophyll,3) if chlorophyll is not None else None,
                "score": score,
                "level": get_habitat_level(score),
                "data_completeness": completeness,
            })

    if not cells:
        raise HTTPException(status_code=503, detail={"message":"Grid oluşturulamadı.","errors":errors})

    return {
        "ok": True,
        "species": {"id":"bluefish","name":"Lüfer","scientific_name":"Pomatomus saltatrix",
                    "life_stage":"adult","color":"#1687FF"},
        "region": "turkey",
        "grid": {"step_degrees":HABITAT_GRID_STEP,"latitude_count":len(latitudes),
                 "longitude_count":len(longitudes),"total_cells":total_cells,"scored_cells":scored_cells},
        "model": {
            "version":"0.1","type":"provisional_temperature_suitability",
            "score_type":"habitat_suitability_not_probability",
            "temperature":{"minimum_modeled_range":8.4,"maximum_modeled_range":27.5,
                           "reported_mean":21.1,"unit":"C"},
            "limitations":["Skor balığın gerçek varlığını veya yakalanma olasılığını göstermez.",
                           "Sıcaklık modeli geçicidir.","Hücre çözünürlüğü yaklaşık 0.25 derecedir."]
        },
        "source_times":   source_times,
        "datasets":       dataset_status,
        "errors":         errors,
        "cells":          cells,
        "cached":         False,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
    }

# ==================================================
# FIRESTORE YARDIMCI FONKSİYONLARI
# ==================================================

CHUNK_SIZE = 100  # Her chunk'ta max hücre sayısı


def save_cells_to_firestore(doc_id, cells):
    """cells dizisini chunk'lara bölerek Firestore sub-collection'a yaz."""
    ref    = _db.collection("habitat_grids").document(doc_id)
    chunks = [cells[i:i+CHUNK_SIZE] for i in range(0, len(cells), CHUNK_SIZE)]
    batch  = _db.batch()
    for idx, chunk in enumerate(chunks):
        chunk_ref = ref.collection("chunks").document(str(idx))
        batch.set(chunk_ref, {"cells": chunk, "index": idx})
    batch.commit()
    return len(chunks)


def load_cells_from_firestore(doc_id):
    """Firestore sub-collection'dan tüm chunk'ları oku, cells listesini döndür."""
    ref    = _db.collection("habitat_grids").document(doc_id)
    chunks = ref.collection("chunks").order_by("index").stream()
    cells  = []
    for chunk in chunks:
        cells.extend(chunk.to_dict().get("cells", []))
    return cells

# ==================================================
# HABITAT GRID API
# ==================================================

@app.get("/habitat-grid")
def habitat_grid(species: str = "bluefish", region: str = "turkey", refresh: bool = False):
    species = species.lower().strip()
    region  = region.lower().strip()
    if species != "bluefish":
        raise HTTPException(status_code=400, detail="Şimdilik yalnızca bluefish destekleniyor.")
    if region != "turkey":
        raise HTTPException(status_code=400, detail="Şimdilik yalnızca turkey bölgesi destekleniyor.")

    cache_key = f"{species}:{region}"
    doc_id    = f"{species}_{region}"

    # 1. Bellek cache
    if not refresh:
        cached = get_habitat_cache(cache_key)
        if cached:
            response = dict(cached)
            response["cached"] = True
            return response

    # 2. Firestore'dan metadata + cells oku
    if not refresh and _db:
        try:
            snap = _db.collection("habitat_grids").document(doc_id).get()
            if snap.exists:
                data  = snap.to_dict()
                cells = load_cells_from_firestore(doc_id)
                if cells:
                    data["cells"]  = cells
                    data["cached"] = True
                    data["source"] = "firestore"
                    set_habitat_cache(cache_key, data)
                    return data
        except Exception as e:
            print(f"Firestore read error: {e}")

    # 3. CMEMS'ten hesapla
    response = build_habitat_response(species, region)
    set_habitat_cache(cache_key, response)
    return response

# ==================================================
# SAVE GRID → Firestore (cron tarafından çağrılır)
# ==================================================

@app.get("/save-grid")
def save_grid(species: str = "bluefish", region: str = "turkey", secret: str = ""):
    if secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _db:
        raise HTTPException(status_code=500, detail="Firebase bağlı değil")

    data   = build_habitat_response(species, region)
    doc_id = f"{species}_{region}"
    ref    = _db.collection("habitat_grids").document(doc_id)

    # Metadata'yı ana belgeye kaydet (cells hariç)
    metadata = {k: v for k, v in data.items() if k != "cells"}
    ref.set(metadata)

    # cells'i chunk'lara bölerek sub-collection'a kaydet
    num_chunks = save_cells_to_firestore(doc_id, data["cells"])

    return {
        "ok":           True,
        "saved_cells":  len(data["cells"]),
        "chunks":       num_chunks,
        "generated_at": data["generated_at"],
        "species":      species,
        "region":       region,
    }
