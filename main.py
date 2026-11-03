
import os
import math
from datetime import datetime, timedelta, timezone

import copernicusmarine

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="OLTA Marine API",
    version="3.0"
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

# Karadeniz - güncel günlük 1 km
CHL_BLACKSEA_NRT = (
    "cmems_obs-oc_blk_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D"
)

# Karadeniz - geçmiş günlük 1 km
CHL_BLACKSEA_HISTORY = (
    "cmems_obs-oc_blk_bgc-plankton_my_l4-gapfree-multi-1km_P1D"
)

# Küresel - günlük 4 km
CHL_GLOBAL = (
    "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
)


# ==================================================
# CHLOROPHYLL SOURCE PRIORITY
# ==================================================

CHL_SOURCES = [
    {
        "name": "blacksea_nrt_1km",
        "id": CHL_BLACKSEA_NRT,
        "regional": True
    },
    {
        "name": "blacksea_history_1km",
        "id": CHL_BLACKSEA_HISTORY,
        "regional": True
    },
    {
        "name": "global_4km",
        "id": CHL_GLOBAL,
        "regional": False
    }
]


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
            f"{variable} bulunamadı: {list(ds.data_vars)}"
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
# CHLOROPHYLL HELPERS
# ==================================================

def is_blacksea_region(lat, lng):

    return (
        40.0 <= lat <= 48.0
        and 26.5 <= lng <= 42.0
    )


def get_chlorophyll_records(
    source,
    lat,
    lng,
    start_date,
    end_date
):

    ds = None

    try:

        ds = get_dataset(
            source["id"],
            ["CHL"],
            lat,
            lng,
            start_date=start_date,
            end_date=end_date
        )

        if "CHL" not in ds:
            raise Exception(
                f"CHL değişkeni bulunamadı: {list(ds.data_vars)}"
            )

        data = ds["CHL"]

        # Koordinat isimlerini otomatik bul
        latitude_name = next(
            (
                name for name in ["latitude", "lat"]
                if name in data.dims
            ),
            None
        )

        longitude_name = next(
            (
                name for name in ["longitude", "lon"]
                if name in data.dims
            ),
            None
        )

        if latitude_name is None or longitude_name is None:
            raise Exception(
                f"Koordinatlar bulunamadı: {list(data.dims)}"
            )

        data = data.sel(
            {
                latitude_name: lat,
                longitude_name: lng
            },
            method="nearest"
        )

        if "time" not in data.dims:
            raise Exception(
                "CHL verisinde zaman boyutu bulunamadı."
            )

        values = data.compute().values
        times = ds["time"].values

        records = {}

        for timestamp, raw_value in zip(times, values):

            date = str(timestamp)[:10]

            value = float(raw_value)

            # Geçersiz ve dolgu değerlerini ele
            if (
                not math.isfinite(value)
                or value <= 0
                or value > 100
            ):
                records[date] = None

            else:
                records[date] = value

        return records, None

    except Exception as e:

        return {}, str(e)[:500]

    finally:

        if ds is not None:
            ds.close()


def get_chlorophyll_history_data(lat, lng, days):

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    end_date = now.strftime("%Y-%m-%dT23:59:59")

    start_date = (
        now - timedelta(days=days)
    ).strftime("%Y-%m-%dT00:00:00")

    merged = {}
    source_status = {}

    # Bölgesel kaynakları yalnızca kapsama alanında dene
    sources = []

    for source in CHL_SOURCES:

        if source["regional"] and not is_blacksea_region(lat, lng):
            continue

        sources.append(source)

    # Öncelik sırasına göre kaynakları birleştir
    for source in sources:

        records, error = get_chlorophyll_records(
            source,
            lat,
            lng,
            start_date,
            end_date
        )

        valid_count = sum(
            1 for value in records.values()
            if value is not None
        )

        source_status[source["name"]] = {
            "ok": error is None,
            "valid_records": valid_count,
            "total_records": len(records),
            "error": error
        }

        for date, value in records.items():

            # Daha öncelikli kaynağın geçerli verisini koru
            if value is not None:

                if date not in merged or merged[date] is None:

                    merged[date] = {
                        "value": value,
                        "source": source["name"]
                    }

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
        "user": bool(CMEMS_USER),
        "version": "3.0"
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
            "variables": ["tob", "sob", "mlotst", "zos"],
            "depth": False
        },
        "currents": {
            "id": CURRENTS_DATASET,
            "variables": ["uo", "vo"],
            "depth": True
        },
        "chlorophyll_blacksea_nrt": {
            "id": CHL_BLACKSEA_NRT,
            "variables": ["CHL"],
            "depth": False
        },
        "chlorophyll_blacksea_history": {
            "id": CHL_BLACKSEA_HISTORY,
            "variables": ["CHL"],
            "depth": False
        },
        "chlorophyll_global": {
            "id": CHL_GLOBAL,
            "variables": ["CHL"],
            "depth": False
        }
    }

    results = {}

    for name, config in datasets.items():

        if (
            name.startswith("chlorophyll_blacksea")
            and not is_blacksea_region(lat, lng)
        ):
            results[name] = {
                "skipped": True,
                "reason": "Koordinat Karadeniz veri alanı dışında."
            }
            continue

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
        "chlorophyll_time": None,
        "chlorophyll_source": None,

        "errors": {}
    }

    # ----------------------------------------------
    # PHYSICAL DATA
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
    # CURRENTS
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

            result["current_speed"] = math.sqrt(
                u ** 2 + v ** 2
            )

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
    # CHLOROPHYLL - AUTOMATIC SOURCE SELECTION
    # ----------------------------------------------

    try:

        history, source_status = get_chlorophyll_history_data(
            lat,
            lng,
            30
        )

        valid_dates = [
            (date, item)
            for date, item in history.items()
            if item is not None
        ]

        result["chlorophyll_sources"] = source_status

        if valid_dates:

            latest_date, latest_item = max(
                valid_dates,
                key=lambda item: item[0]
            )

            result["chlorophyll"] = latest_item["value"]
            result["chlorophyll_time"] = latest_date
            result["chlorophyll_source"] = latest_item["source"]

        else:

            result["errors"]["chlorophyll"] = (
                "Denemeye alınan kaynaklarda geçerli ölçüm bulunamadı."
            )

    except Exception as e:

        result["errors"]["chlorophyll"] = str(e)[:500]

    # ----------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------

    if result["errors"]:
        result["status"] = "partial"
    else:
        result["status"] = "ok"

    return result


# ==================================================
# CHLOROPHYLL HISTORY API
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

    history, source_status = get_chlorophyll_history_data(
        lat,
        lng,
        days
    )

    records = []

    for date in sorted(history.keys()):

        item = history[date]

        if item is None:

            records.append({
                "date": date,
                "chlorophyll": None,
                "source": None
            })

        else:

            records.append({
                "date": date,
                "chlorophyll": item["value"],
                "source": item["source"]
            })

    valid_values = [
        record["chlorophyll"]
        for record in records
        if record["chlorophyll"] is not None
    ]

    return {
        "ok": True,
        "lat": lat,
        "lng": lng,
        "requested_days": days,

        "total_records": len(records),
        "valid_records": len(valid_values),
        "missing_records": len(records) - len(valid_values),

        "average_chlorophyll": (
            sum(valid_values) / len(valid_values)
            if valid_values else None
        ),

        "maximum_chlorophyll": (
            max(valid_values) if valid_values else None
        ),

        "minimum_chlorophyll": (
            min(valid_values) if valid_values else None
        ),

        "source_status": source_status,
        "records": records
    }
