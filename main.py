import os
import hmac
import json
import math
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import copernicusmarine
import firebase_admin
from firebase_admin import credentials, firestore as fs

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(title="OLTA Marine API", version="3.7")

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

# Varsayılan yok: tanımlı değilse korumalı uç noktalar kapalıdır.
CRON_SECRET = os.environ.get("CRON_SECRET") or None


def is_authorized(secret_value):
    if not CRON_SECRET or not secret_value:
        return False
    return hmac.compare_digest(str(secret_value), CRON_SECRET)

# ==================================================
# COPERNICUS CREDENTIALS
# ==================================================

CMEMS_USER = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS = os.environ.get("CMEMS_PASSWORD")

# ==================================================
# DATASET IDS
# ==================================================

# Yüzey sıcaklığı / tuzluluğu (3D, depth=0 ≈ 0.5 m)
THETAO_DATASET   = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
SO_DATASET       = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
# 2D fizik: tob/sob = DİP sıcaklığı/tuzluluğu, mlotst, zos
PHYSICS_DATASET  = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
CURRENTS_DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"

CHL_BLACKSEA_NRT     = "cmems_obs-oc_blk_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D"
CHL_BLACKSEA_HISTORY = "cmems_obs-oc_blk_bgc-plankton_my_l4-gapfree-multi-1km_P1D"
CHL_GLOBAL_NRT       = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"
CHL_GLOBAL           = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"

BGC_BIO_DATASET      = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
BGC_NUT_DATASET      = "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m"
BGC_PFT_DATASET      = "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m"
BGC_PLANKTON_DATASET = "cmems_mod_glo_bgc-plankton_anfc_0.25deg_P1D-m"

CHL_SOURCES = [
    {"name": "blacksea_nrt_1km",     "id": CHL_BLACKSEA_NRT,     "regional": True},
    {"name": "blacksea_history_1km", "id": CHL_BLACKSEA_HISTORY,  "regional": True},
    {"name": "global_nrt_4km",       "id": CHL_GLOBAL_NRT,        "regional": False},
    {"name": "global_history_4km",   "id": CHL_GLOBAL,            "regional": False},
]

# ==================================================
# COMMON FUNCTIONS
# ==================================================

def validate_coordinates(lat, lng):
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Geçersiz enlem")
    if not (-180 <= lng <= 180):
        raise HTTPException(status_code=400, detail="Geçersiz boylam")


def utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def default_time_window():
    """Analiz/tahmin setlerinde bugünü kapsayan dar pencere (indirmeyi küçültür)."""
    now = utc_now_naive()
    return ((now - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00"),
            (now + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59"))


def select_current_time(data):
    """'anfc' setlerinin son adımı ~10 gün sonrası tahmindir; bugüne en yakın adımı seç."""
    if "time" in data.dims:
        # Günlük ortalamalar 00:00 damgalı: öğleden sonra "en yakın" yarın olmasın diye günün başı
        today = utc_now_naive().replace(hour=0, minute=0, second=0, microsecond=0)
        data = data.sortby("time").sel(time=np.datetime64(today), method="nearest")
    return data


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
    if not start_date and not end_date:
        start_date, end_date = default_time_window()
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
    data = select_current_time(data)
    if "depth" in data.dims: data = data.sortby("depth").isel(depth=0)
    value = float(np.asarray(data.compute().values).squeeze())
    return None if not math.isfinite(value) else value


def get_dataset_time(ds):
    if "time" not in ds.coords:
        return None
    try:
        return str(select_current_time(ds["time"]).values)
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


CHL_SEARCH_RADIUS_KM = 10      # kıyıda en yakın geçerli uydu pikselini bu yarıçapta ara
CHL_SEARCH_BUFFER    = 0.12    # derece; ~10 km yarıçapı kapsayan indirme kutusu
CHL_FRESH_DAYS       = 3       # bu kadar yeni ölçüm bulununca sonraki kaynaklara geçme


def get_chlorophyll_records(source, lat, lng, start_date, end_date):
    """Her gün için noktaya en yakın GEÇERLİ pikseli döndürür.

    Kıyıda/boğazda en yakın piksel genelde kara veya bulut maskesidir; bu yüzden
    yalnızca en yakın piksele bakmak yerine CHL_SEARCH_RADIUS_KM içindeki en
    yakın geçerli ölçüm alınır.
    """
    ds = None
    try:
        ds = get_dataset(source["id"], ["CHL"], lat, lng,
                         start_date=start_date, end_date=end_date,
                         buffer=CHL_SEARCH_BUFFER)
        if "CHL" not in ds:
            raise Exception(f"CHL bulunamadı: {list(ds.data_vars)}")
        data = ds["CHL"]
        lat_name = next((n for n in ["latitude","lat"] if n in data.dims), None)
        lng_name = next((n for n in ["longitude","lon"] if n in data.dims), None)
        if not lat_name or not lng_name:
            raise Exception(f"Koordinatlar bulunamadı: {list(data.dims)}")
        if "time" not in data.dims:
            raise Exception("CHL verisinde zaman boyutu bulunamadı.")
        data = data.transpose("time", lat_name, lng_name, ...).squeeze(drop=True)
        if data.ndim == 2:                       # tek zaman adımı
            data = data.expand_dims("time")
        values = np.asarray(data.compute().values, dtype=float)
        times  = np.asarray(data["time"].values).reshape(-1)
        lats   = np.asarray(data[lat_name].values, dtype=float)
        lngs   = np.asarray(data[lng_name].values, dtype=float)

        # Noktaya mesafe (km, eşdikdörtgen yaklaşımı — 10 km ölçeğinde yeterli)
        dlat = (lats[:, None] - lat) * 111.32
        dlng = (lngs[None, :] - lng) * 111.32 * math.cos(math.radians(lat))
        dist = np.sqrt(dlat**2 + dlng**2)
        in_radius = dist <= CHL_SEARCH_RADIUS_KM

        records = {}
        for t_idx, timestamp in enumerate(times):
            date  = str(timestamp)[:10]
            frame = values[t_idx]
            valid = np.isfinite(frame) & (frame > 0) & (frame <= 100) & in_radius
            if not valid.any():
                records[date] = None
                continue
            masked = np.where(valid, dist, np.inf)
            k = np.unravel_index(np.argmin(masked), masked.shape)
            records[date] = {"value": float(frame[k]), "distance_km": round(float(dist[k]), 2)}
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
    fresh_since = (now - timedelta(days=CHL_FRESH_DAYS)).strftime("%Y-%m-%d")
    merged = {}
    source_status = {}
    sources = [s for s in CHL_SOURCES
               if not (s["regional"] and not is_blacksea_region(lat, lng))]
    for source in sources:
        # Yeterince güncel ölçüm varsa kalan kaynakları indirme (bellek/süre)
        if any(v is not None and d >= fresh_since for d, v in merged.items()):
            source_status[source["name"]] = {"ok": None, "skipped": True}
            continue
        records, error = get_chlorophyll_records(source, lat, lng, start_date, end_date)
        valid_count = sum(1 for v in records.values() if v is not None)
        source_status[source["name"]] = {
            "ok": error is None, "valid_records": valid_count,
            "total_records": len(records), "error": error
        }
        for date, rec in records.items():
            if rec is not None:
                if date not in merged or merged[date] is None:
                    merged[date] = {**rec, "source": source["name"]}
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
        "version":  "3.7"
    }

# ==================================================
# MARINE API
# ==================================================

MARINE_CACHE_TTL = 60 * 60
_marine_cache      = {}
_marine_cache_lock = threading.Lock()


def read_surface_point(result, key, dataset_id, variable, fallback_variable, lat, lng):
    """Yüzey değerini oku; set açılamazsa 2D fizik setindeki dip değerine düş."""
    ds = None
    try:
        ds = get_dataset(dataset_id, [variable], lat, lng, depth=True)
        result[key] = get_value(ds, variable, lat, lng)
        result[f"{key}_source"] = "surface"
        result[f"{key}_time"]   = get_dataset_time(ds)
        return
    except Exception as e:
        result["errors"][key] = str(e)[:500]
    finally:
        if ds: ds.close()
    ds = None
    try:
        ds = get_dataset(PHYSICS_DATASET, [fallback_variable], lat, lng)
        result[key] = get_value(ds, fallback_variable, lat, lng)
        result[f"{key}_source"] = "bottom_fallback"
        result[f"{key}_time"]   = get_dataset_time(ds)
    except Exception as e:
        result["errors"][f"{key}_fallback"] = str(e)[:500]
    finally:
        if ds: ds.close()


@app.get("/marine")
def marine(lat: float, lng: float):
    validate_coordinates(lat, lng)
    cache_key = (round(lat / 0.05) * 0.05, round(lng / 0.05) * 0.05)
    with _marine_cache_lock:
        item = _marine_cache.get(cache_key)
        if item and time.time() - item["created"] < MARINE_CACHE_TTL:
            return {**item["data"], "lat": lat, "lng": lng, "cached": True}
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
        ds = get_dataset(PHYSICS_DATASET, ["mlotst","zos"], lat, lng)
        result["mixing_layer_depth"] = get_value(ds, "mlotst", lat, lng)
        result["sea_level"]          = get_value(ds, "zos",    lat, lng)
        result["physics_time"]       = get_dataset_time(ds)
    except Exception as e:
        result["errors"]["physics"] = str(e)[:500]
    finally:
        if ds: ds.close()
    read_surface_point(result, "temperature", THETAO_DATASET, "thetao", "tob", lat, lng)
    read_surface_point(result, "salinity",    SO_DATASET,     "so",     "sob", lat, lng)
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
            result["chlorophyll_distance_km"] = latest_item.get("distance_km")
        else:
            result["errors"]["chlorophyll"] = "Geçerli klorofil ölçümü bulunamadı."
    except Exception as e:
        result["errors"]["chlorophyll"] = str(e)[:500]
    result["status"] = "partial" if result["errors"] else "ok"
    with _marine_cache_lock:
        _marine_cache[cache_key] = {"created": time.time(), "data": result}
    return result

# ==================================================
# HABITAT GRID CONFIG
# ==================================================

HABITAT_GRID_STEP = 0.1
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
    params["start_datetime"], params["end_datetime"] = default_time_window()
    return copernicusmarine.open_dataset(**params)


def regional_data_to_grid(ds, variable, target_latitudes, target_longitudes):
    if variable not in ds:
        raise Exception(f"{variable} değişkeni bulunamadı.")
    lat_name = get_coordinate_name(ds, ["latitude","lat"])
    lng_name = get_coordinate_name(ds, ["longitude","lon"])
    if not lat_name or not lng_name:
        raise Exception("Enlem/boylam koordinatları bulunamadı.")
    data = ds[variable]
    data = select_current_time(data)
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

    # (çıktı anahtarı, [(dataset, değişken, depth), yedekler...])
    layer_specs = [
        ("temperature", [(THETAO_DATASET, "thetao", True), (PHYSICS_DATASET, "tob", False)]),
        ("salinity",    [(SO_DATASET,     "so",     True), (PHYSICS_DATASET, "sob", False)]),
        ("oxygen",      [(BGC_BIO_DATASET, "o2",    True)]),
        ("chlorophyll", [(BGC_PFT_DATASET, "chl",   True)]),
    ]
    for key, candidates in layer_specs:
        attempt_errors = []
        for ds_id, var, depth_ in candidates:
            ds = None
            try:
                ds = get_regional_dataset(ds_id, [var], depth=depth_)
                arrays[key]         = regional_data_to_grid(ds, var, latitudes, longitudes)
                source_times[key]   = get_dataset_time(ds)
                dataset_status[key] = {"ok": True, "dataset_id": ds_id, "variable": var,
                                       "fallback": var in ("tob", "sob")}
                break
            except Exception as e:
                attempt_errors.append(f"{ds_id}/{var}: {str(e)[:300]}")
            finally:
                if ds: ds.close()
        if arrays[key] is None:
            errors[key]         = " | ".join(attempt_errors)[:1000]
            dataset_status[key] = {"ok": False, "error": errors[key]}
        elif attempt_errors:
            errors[f"{key}_primary"] = " | ".join(attempt_errors)[:1000]

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
            "version":"0.2","type":"provisional_temperature_suitability",
            "score_type":"habitat_suitability_not_probability",
            "temperature":{"minimum_modeled_range":8.4,"maximum_modeled_range":27.5,
                           "reported_mean":21.1,"unit":"C"},
            "limitations":[
                "Skor balığın gerçek varlığını veya yakalanma olasılığını göstermez.",
                "Çevresel uygunluk göstergesidir, av garantisi değildir.",
                "Sıcaklık ve tuzluluk modeli geçicidir.",
                "Değerler yüzeye en yakın model katmanından (~0.5 m), bugüne en yakın günden alınır.",
                f"Hücre çözünürlüğü yaklaşık {HABITAT_GRID_STEP} derecedir.",
            ]
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

CHUNK_SIZE = 500  # Her chunk'ta max hücre sayısı (~9 chunk, 42 yerine)


def delete_old_chunks(ref, keep_generation_id=None):
    """Eski generation_id'li chunk'ları sil. keep_generation_id verilen nesli koru."""
    old   = ref.collection("chunks").stream()
    batch = _db.batch()
    count = 0
    for doc in old:
        if keep_generation_id and doc.to_dict().get("generation_id") == keep_generation_id:
            continue  # Yeni nesli silme
        batch.delete(doc.reference)
        count += 1
        if count >= 490:
            batch.commit()
            batch = _db.batch()
            count = 0
    if count > 0:
        batch.commit()


def save_cells_to_firestore(ref, cells, generation_id):
    """cells dizisini chunk'lara bölerek Firestore sub-collection'a yaz."""
    chunks = [cells[i:i+CHUNK_SIZE] for i in range(0, len(cells), CHUNK_SIZE)]
    batch  = _db.batch()
    for idx, chunk in enumerate(chunks):
        # Nesle özel ID: yeni yazım eski nesli ezmez, okuyucular tutarlı kalır.
        chunk_ref = ref.collection("chunks").document(f"{generation_id}_{idx:03d}")
        batch.set(chunk_ref, {
            "cells":         chunk,
            "index":         idx,
            "generation_id": generation_id,
        })
    batch.commit()
    return len(chunks)


def load_cells_from_firestore(doc_id, generation_id=None):
    """Firestore sub-collection'dan tüm chunk'ları oku."""
    ref    = _db.collection("habitat_grids").document(doc_id)
    chunks = ref.collection("chunks").order_by("index").stream()
    cells  = []
    for chunk in chunks:
        d = chunk.to_dict()
        # Generation ID kontrolü — eski chunk'ları atla
        if generation_id and d.get("generation_id") != generation_id:
            continue
        cells.extend(d.get("cells", []))
    return cells




# ==================================================
# HABITAT GRID API
# ==================================================

_habitat_build_lock = threading.Lock()


@app.get("/habitat-grid")
def habitat_grid(species: str = "bluefish", region: str = "turkey", refresh: bool = False,
                 x_cron_secret: Optional[str] = Header(default=None)):
    species = species.lower().strip()
    region  = region.lower().strip()
    if species != "bluefish":
        raise HTTPException(status_code=400, detail="Şimdilik yalnızca bluefish destekleniyor.")
    if region != "turkey":
        raise HTTPException(status_code=400, detail="Şimdilik yalnızca turkey bölgesi destekleniyor.")

    cache_key = f"{species}:{region}"
    doc_id    = f"{species}_{region}"

    # refresh pahalı CMEMS hesabını tetikler; yalnızca yetkili çağrılara izin ver
    if refresh and not is_authorized(x_cron_secret):
        refresh = False

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
                data          = snap.to_dict()
                generation_id = data.get("generation_id")
                cells         = load_cells_from_firestore(doc_id, generation_id)
                if cells:
                    data["cells"]  = cells
                    data["cached"] = True
                    data["source"] = "firestore"
                    set_habitat_cache(cache_key, data)
                    return data
        except Exception as e:
            print(f"Firestore read error: {e}")

    # 3. CMEMS'ten hesapla — aynı anda tek hesaplama
    with _habitat_build_lock:
        if not refresh:
            cached = get_habitat_cache(cache_key)
            if cached:
                return {**cached, "cached": True}
        response = build_habitat_response(species, region)
        set_habitat_cache(cache_key, response)
    return response

# ==================================================
# SAVE GRID → Firestore (cron tarafından çağrılır)
# ==================================================

@app.get("/save-grid")
def save_grid(species: str = "bluefish", region: str = "turkey", secret: str = "",
              x_cron_secret: Optional[str] = Header(default=None)):
    if not CRON_SECRET:
        raise HTTPException(status_code=503, detail="CRON_SECRET tanımlı değil")
    # Tercih edilen: X-Cron-Secret header. ?secret= eski cron'lar için geçici olarak destekleniyor.
    if not is_authorized(x_cron_secret or secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _db:
        raise HTTPException(status_code=500, detail="Firebase bağlı değil")

    with _habitat_build_lock:
        data = build_habitat_response(species, region)
    set_habitat_cache(f"{species}:{region}", data)
    doc_id        = f"{species}_{region}"
    generation_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    ref           = _db.collection("habitat_grids").document(doc_id)

    # 1. Yeni chunk'ları yaz (generation_id ile işaretli)
    num_chunks = save_cells_to_firestore(ref, data["cells"], generation_id)

    # 2. Metadata'yı en son güncelle (chunk'lar hazır olduktan sonra)
    metadata = {k: v for k, v in data.items() if k != "cells"}
    metadata["generation_id"] = generation_id
    ref.set(metadata)

    # 3. Eski nesil chunk'ları temizle (yeni nesli koru)
    delete_old_chunks(ref, keep_generation_id=generation_id)

    return {
        "ok":           True,
        "saved_cells":  len(data["cells"]),
        "chunks":       num_chunks,
        "generated_at": data["generated_at"],
        "species":      species,
        "region":       region,
    }
