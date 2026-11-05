"""Ana sayfa akışı: günlük "en uygun kıyılar" gönderileri.

Her gün grid hesaplandıktan sonra, yasak dönemde olmayan türlerden birkaçı
için kıyıya bitişik deniz kareleri arasında uygunluk skoru en yüksek
noktalar seçilir ve şablon cümlelerle bir gönderi yazılır. Yapay zekâ
kullanılmaz; metin yalnızca verideki sayılardan üretilir.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

import numpy as np

POSTS_PER_DAY      = 4      # günde en fazla kaç tür paylaşılır
SPOTS_PER_POST     = 3      # bir gönderide en fazla kaç kıyı noktası
MIN_SPOT_SCORE     = 50     # bundan düşük skorlu nokta paylaşılmaz
MAX_COAST_KM       = 9.0    # kare merkezi kıyıdan en fazla bu kadar uzak (≈ bir kare)
MAX_PLACE_KM       = 12.0   # kıyı noktası bilinen bir Türkiye kıyı yerine bu kadar yakın olmalı
MIN_SPOT_SEPARATION_KM = 25.0
SEAWARD_OFFSET_KM  = 0.3    # pin kıyı çizgisinin hemen denize tarafına konur
STRONG_CURRENT_MS  = 0.5

KM_PER_DEG = 111.32

# ---------------------------------------------------------------------------
# Türkiye kıyı yerleri (yaklaşık koordinatlar). Hem yer adı vermek hem de
# komşu ülke kıyılarındaki (Yunan adaları, Kıbrıs vb.) noktaları elemek için.
# ---------------------------------------------------------------------------
COASTAL_PLACES = [
    # Karadeniz
    ("İğneada", 41.875, 27.98), ("Kıyıköy", 41.63, 28.09), ("Karaburun", 41.35, 28.68),
    ("Kilyos", 41.25, 29.03), ("Rumeli Feneri", 41.235, 29.11), ("Riva", 41.22, 29.22),
    ("Şile", 41.18, 29.61), ("Ağva", 41.14, 29.86), ("Kefken", 41.17, 30.23),
    ("Karasu", 41.10, 30.69), ("Akçakoca", 41.09, 31.12), ("Karadeniz Ereğli", 41.28, 31.42),
    ("Zonguldak", 41.455, 31.79), ("Amasra", 41.75, 32.39), ("Cide", 41.89, 33.0),
    ("İnebolu", 41.98, 33.76), ("Abana", 41.98, 34.01), ("Sinop", 42.03, 35.15),
    ("Gerze", 41.80, 35.20), ("Bafra", 41.72, 35.95), ("Samsun", 41.29, 36.33),
    ("Terme", 41.21, 36.97), ("Ünye", 41.13, 37.29), ("Fatsa", 41.03, 37.50),
    ("Ordu", 40.98, 37.88), ("Giresun", 40.92, 38.39), ("Tirebolu", 41.01, 38.82),
    ("Görele", 41.03, 39.00), ("Vakfıkebir", 41.05, 39.28), ("Akçaabat", 41.02, 39.57),
    ("Trabzon", 41.00, 39.72), ("Sürmene", 40.91, 40.11), ("Of", 40.95, 40.27),
    ("Rize", 41.02, 40.52), ("Pazar", 41.18, 40.88), ("Ardeşen", 41.19, 41.00),
    ("Fındıklı", 41.27, 41.14), ("Hopa", 41.39, 41.43),
    # İstanbul Boğazı ve Marmara
    ("Sarıyer", 41.17, 29.06), ("Beşiktaş", 41.04, 29.00), ("Kadıköy", 40.99, 29.03),
    ("Bostancı", 40.95, 29.09), ("Adalar", 40.87, 29.12), ("Kartal", 40.89, 29.19),
    ("Tuzla", 40.82, 29.30), ("Florya", 40.97, 28.79), ("Silivri", 41.07, 28.25),
    ("Marmara Ereğlisi", 40.97, 27.95), ("Tekirdağ", 40.98, 27.51), ("Şarköy", 40.61, 27.11),
    ("Gelibolu", 40.41, 26.67), ("Lapseki", 40.34, 26.69), ("Çanakkale", 40.15, 26.41),
    ("Darıca", 40.76, 29.38), ("İzmit", 40.76, 29.92), ("Gölcük", 40.72, 29.82),
    ("Yalova", 40.66, 29.27), ("Armutlu", 40.52, 28.83), ("Gemlik", 40.43, 29.15),
    ("Mudanya", 40.38, 28.88), ("Bandırma", 40.35, 27.97), ("Erdek", 40.40, 27.79),
    ("Marmara Adası", 40.62, 27.62), ("Karabiga", 40.40, 27.31),
    # Ege
    ("Gökçeada", 40.20, 25.90), ("Bozcaada", 39.83, 26.07), ("Geyikli", 39.80, 26.20),
    ("Babakale", 39.48, 26.07), ("Assos", 39.49, 26.34), ("Küçükkuyu", 39.55, 26.61),
    ("Akçay", 39.58, 26.92), ("Ören", 39.50, 26.95), ("Ayvalık", 39.32, 26.69),
    ("Dikili", 39.07, 26.89), ("Çandarlı", 38.94, 26.94), ("Foça", 38.67, 26.75),
    ("İzmir", 38.46, 27.11), ("Urla", 38.37, 26.77), ("Karaburun Yarımadası", 38.64, 26.51),
    ("Çeşme", 38.32, 26.30), ("Seferihisar", 38.20, 26.79), ("Gümüldür", 38.07, 27.00),
    ("Kuşadası", 37.86, 27.26), ("Didim", 37.37, 27.27), ("Güllük", 37.24, 27.60),
    ("Bodrum", 37.03, 27.43), ("Turgutreis", 37.00, 27.26), ("Datça", 36.73, 27.69),
    ("Marmaris", 36.85, 28.27), ("Akyaka", 37.05, 28.33), ("Dalyan", 36.79, 28.62),
    ("Göcek", 36.75, 28.94), ("Fethiye", 36.65, 29.12), ("Ölüdeniz", 36.55, 29.12),
    # Akdeniz
    ("Kalkan", 36.26, 29.41), ("Kaş", 36.20, 29.64), ("Kekova", 36.20, 29.85),
    ("Demre", 36.20, 29.99), ("Finike", 36.30, 30.15), ("Kumluca", 36.30, 30.35),
    ("Adrasan", 36.31, 30.48), ("Kemer", 36.60, 30.56), ("Antalya", 36.88, 30.70),
    ("Lara", 36.85, 30.83), ("Belek", 36.85, 31.06), ("Side", 36.77, 31.39),
    ("Manavgat", 36.75, 31.45), ("Alanya", 36.54, 32.00), ("Gazipaşa", 36.27, 32.31),
    ("Anamur", 36.07, 32.85), ("Bozyazı", 36.10, 32.97), ("Aydıncık", 36.14, 33.32),
    ("Taşucu", 36.32, 33.88), ("Susanoğlu", 36.36, 34.06), ("Erdemli", 36.60, 34.31),
    ("Mersin", 36.78, 34.62), ("Kazanlı", 36.82, 34.80), ("Karataş", 36.57, 35.37),
    ("Yumurtalık", 36.77, 35.79), ("Dörtyol", 36.86, 36.16), ("İskenderun", 36.59, 36.17),
    ("Arsuz", 36.41, 35.89), ("Samandağ", 36.07, 35.93),
]

MONTHS = {"Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5, "Haziran": 6,
          "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12}

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def tr_lower(text):
    return (text or "").replace("I", "ı").replace("İ", "i").lower()


def tr_num(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def tr_join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " ve " + items[-1]


def slugify(text):
    table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    s = (text or "").translate(table).lower()
    return "".join(ch if ch.isalnum() else "-" for ch in s).strip("-")


def km_between(lat1, lng1, lat2, lng2):
    dy = (lat1 - lat2) * KM_PER_DEG
    dx = (lng1 - lng2) * KM_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def is_banned(fish, today):
    """species.html'deki isBanned ile aynı: banStart/banEnd "15 Nisan" biçiminde."""
    start, end = fish.get("banStart"), fish.get("banEnd")
    if not start or not end:
        return False
    try:
        sd, sm = start.split()
        ed, em = end.split()
        s = (MONTHS[sm], int(sd))
        e = (MONTHS[em], int(ed))
        t = (today.month, today.day)
        return (t >= s or t <= e) if e < s else (s <= t <= e)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Kıyı çizgisi (kara poligonlarının tüm halkaları: Karadeniz/Marmara bu
# veride ana kara poligonunun iç halkaları olarak duruyor)
# ---------------------------------------------------------------------------
_coast_segments = None


def coast_segments():
    global _coast_segments
    if _coast_segments is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "land-tr.geojson")
        with open(path, encoding="utf-8") as f:
            land = json.load(f)
        a, b = [], []
        for feat in land.get("features", []):
            geom = feat.get("geometry") or {}
            polys = [geom["coordinates"]] if geom.get("type") == "Polygon" else geom.get("coordinates", [])
            for poly in polys:
                for ring in poly:
                    pts = np.asarray(ring, dtype=float)
                    if len(pts) < 2:
                        continue
                    a.append(pts[:-1]); b.append(pts[1:])
        _coast_segments = (np.concatenate(a), np.concatenate(b))   # (N,2) lng,lat
    return _coast_segments


def nearest_coast_point(lat, lng):
    """(kıyı_lat, kıyı_lng, mesafe_km). Yerel eşdikdörtgen düzlemde nokta-doğru parçası."""
    a, b = coast_segments()
    k = math.cos(math.radians(lat))
    ax, ay = (a[:, 0] - lng) * k, a[:, 1] - lat
    bx, by = (b[:, 0] - lng) * k, b[:, 1] - lat
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    t = np.clip(-(ax * dx + ay * dy) / np.where(seg2 > 0, seg2, 1), 0, 1)
    px, py = ax + t * dx, ay + t * dy
    d2 = px * px + py * py
    i = int(np.argmin(d2))
    return lat + py[i], lng + px[i] / k, math.sqrt(d2[i]) * KM_PER_DEG


def nearest_place(lat, lng):
    best, best_km = None, float("inf")
    for name, plat, plng in COASTAL_PLACES:
        d = km_between(lat, lng, plat, plng)
        if d < best_km:
            best, best_km = name, d
    return best, best_km

# ---------------------------------------------------------------------------
# Kıyı noktaları
# ---------------------------------------------------------------------------

def coastal_candidates(cells, step=0.1):
    """Kıyıya bitişik deniz karelerini, kıyı noktası ve yer adıyla döndür.

    Bir kare, 8 komşusundan biri bölge içinde olup denizde değilse (grid'de
    yoksa) kıyı karesidir. Tür bağımsızdır; gün içinde bir kez hesaplanır.
    """
    sea = {}
    for c in cells:
        if c.get("temperature") is None:
            continue
        if c.get("salinity") is not None and c["salinity"] < 1:
            continue
        sea[(round(c["lat"] / step), round(c["lng"] / step))] = c
    if not sea:
        return []
    li_min = min(k[0] for k in sea); li_max = max(k[0] for k in sea)
    gi_min = min(k[1] for k in sea); gi_max = max(k[1] for k in sea)

    out = []
    for (li, gi), c in sea.items():
        coastal = False
        for dl in (-1, 0, 1):
            for dg in (-1, 0, 1):
                if dl == dg == 0:
                    continue
                n = (li + dl, gi + dg)
                inside = li_min <= n[0] <= li_max and gi_min <= n[1] <= gi_max
                if inside and n not in sea:
                    coastal = True
        if not coastal:
            continue
        clat, clng, coast_km = nearest_coast_point(c["lat"], c["lng"])
        if coast_km > MAX_COAST_KM:
            continue
        place, place_km = nearest_place(clat, clng)
        if place_km > MAX_PLACE_KM:
            continue            # Türkiye kıyısı değil (komşu ülke adası vb.)
        # Pin: kıyı çizgisinden kare merkezine doğru biraz denize
        vy = (c["lat"] - clat) * KM_PER_DEG
        vx = (c["lng"] - clng) * KM_PER_DEG * math.cos(math.radians(clat))
        norm = math.hypot(vx, vy) or 1.0
        f = min(SEAWARD_OFFSET_KM, coast_km) / norm
        pin_lat = clat + vy * f / KM_PER_DEG
        pin_lng = clng + vx * f / (KM_PER_DEG * math.cos(math.radians(clat)))
        out.append({"cell": c, "place": place,
                    "lat": round(pin_lat, 5), "lng": round(pin_lng, 5)})
    return out


def pick_spots(candidates, fish, score_fn):
    scored = []
    for cand in candidates:
        c = cand["cell"]
        s = score_fn(c.get("temperature"), c.get("salinity"), fish)
        if s is not None and s >= MIN_SPOT_SCORE:
            scored.append((s, cand))
    scored.sort(key=lambda x: (-x[0], x[1]["place"]))
    chosen = []
    for s, cand in scored:
        if any(cand["place"] == o["place"] or
               km_between(cand["lat"], cand["lng"], o["lat"], o["lng"]) < MIN_SPOT_SEPARATION_KM
               for o in chosen):
            continue
        c = cand["cell"]
        u, v = c.get("current_u"), c.get("current_v")
        chosen.append({
            "place": cand["place"],
            "label": f"{cand['place']} kıyıları",
            "lat": cand["lat"], "lng": cand["lng"],
            # Skorun hesaplandığı deniz karesi (harita aynı kareyi kullansın)
            "cell_lat": c["lat"], "cell_lng": c["lng"],
            "score": int(s),
            "temperature": c.get("temperature"),
            "salinity": c.get("salinity"),
            "current_speed": round(math.hypot(u, v), 3) if u is not None and v is not None else None,
        })
        if len(chosen) >= SPOTS_PER_POST:
            break
    return chosen

# ---------------------------------------------------------------------------
# Metin (3 cümle, veriye göre değişen şablonlar)
# ---------------------------------------------------------------------------

def compose_text(fish, spots, rng, range_score_fn):
    name  = fish["name"]
    low   = tr_lower(name)
    places = tr_join([s["place"] for s in spots])
    best  = spots[0]

    if len(spots) > 1:
        s1 = rng.choice([
            f"Bugün {low} için koşulların en uygun göründüğü kıyılar: {places}.",
            f"{places} kıyıları bugün {low} için öne çıkıyor.",
            f"{name} arayanlar için bugün {places} kıyıları dikkat çekiyor.",
        ])
    else:
        s1 = rng.choice([
            f"Bugün {low} için koşulların en uygun göründüğü kıyı: {places}.",
            f"{places} kıyıları bugün {low} için öne çıkıyor.",
            f"{name} arayanlar için bugünün adresi {places} kıyıları görünüyor.",
        ])

    t = best.get("temperature")
    s = best.get("salinity")
    s2 = ""
    if t is not None:
        lo, hi = fish["sstMin"], fish["sstMax"]
        rel = (t - lo) / (hi - lo) if hi > lo else 0.5
        if rel < 0.3:
            pos = f"{low} için alt sınıra yakın"
        elif rel > 0.7:
            pos = f"{low} için üst sınıra yakın"
        else:
            pos = rng.choice([f"{low} için ideal aralıkta", f"{low} için tam uygun aralıkta"])
        sal = ""
        if s is not None and fish.get("sssMin") is not None and fish.get("sssMax") is not None:
            ok = (range_score_fn(s, fish["sssMin"], fish["sssMax"]) or 0) > 0
            sal = f", tuzluluk {tr_num(s)} ile {'uygun' if ok else 'aralığın dışında'}"
        s2 = rng.choice([
            f"{best['place']} çevresinde su {tr_num(t)} °C ile {pos}{sal}.",
            f"{best['place']} açığında yüzey suyu {tr_num(t)} °C; bu {pos}{sal}.",
        ])

    cur = best.get("current_speed")
    if cur is not None and cur >= STRONG_CURRENT_MS:
        s3 = f"Akıntı {tr_num(cur)} m/s ile güçlü, takımını buna göre hazırla."
    else:
        s3 = rng.choice([
            "Akıntı sakin; koşullar denemeye değer görünüyor.",
            "Deniz verileri umut verici, rastgele!",
            "Bu değerlendirme deniz verilerine dayanır, av garantisi değildir.",
        ])
    return " ".join(x for x in (s1, s2, s3) if x)

# ---------------------------------------------------------------------------
# Günlük gönderi üretimi
# ---------------------------------------------------------------------------

def build_daily_posts(cells, fish_list, last_posted, today, score_fn, range_score_fn,
                      posts_per_day=POSTS_PER_DAY):
    """Firestore'a yazılacak gönderileri döndürür (yazmaz; test edilebilir)."""
    candidates = coastal_candidates(cells)
    date_str = today.strftime("%Y-%m-%d")
    options = []
    for fish in fish_list:
        if fish.get("type") != "deniz" or fish.get("sstMin") is None or fish.get("sstMax") is None:
            continue
        if is_banned(fish, today):
            continue
        spots = pick_spots(candidates, fish, score_fn)
        if not spots:
            continue
        options.append((last_posted.get(fish["name"], ""), -spots[0]["score"], fish, spots))
    # En uzun süredir paylaşılmayan türler önce; eşitse skoru yüksek olan
    options.sort(key=lambda x: (x[0], x[1], x[2]["name"]))
    posts = []
    for _, _, fish, spots in options[:posts_per_day]:
        rng = random.Random(f"{date_str}:{fish['name']}")
        posts.append({
            "id": f"{date_str.replace('-', '')}_{slugify(fish['name'])}",
            "type": "spot_forecast",
            "species": fish["name"],
            "date": date_str,
            "text": compose_text(fish, spots, rng, range_score_fn),
            "spots": spots,
            "author": {"name": "OLTA", "system": True},
        })
    return posts, len(candidates)
