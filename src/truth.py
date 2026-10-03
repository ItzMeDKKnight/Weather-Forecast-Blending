"""
Ground Truth Provider interface and ERA5 implementation.
Modular interface allows plug-and-play replacement with IMD gridded data.
Day boundary: Asia/Kolkata (IST) calendar day (00:00 to 23:00 IST).
"""
import os, json, time, hashlib, urllib.request, urllib.parse
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class GroundTruthProvider(ABC):
    @abstractmethod
    def fetch_daily(self, points: dict, start_date: str, end_date: str, cache_dir: str = "data/cache") -> pd.DataFrame:
        """Returns DataFrame with [date, point, lat, lon, variable, truth]"""
        pass

def cached_get_json(url: str, params: dict, cache_dir: str = "data/cache", retries: int = 4) -> any:
    os.makedirs(cache_dir, exist_ok=True)
    query_str = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_str}"
    key = hashlib.sha256(full_url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": "SIH26081-ForecastBlend/1.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch {full_url}: {e}")
            time.sleep(2 ** attempt)

class ERA5Provider(GroundTruthProvider):
    ENDPOINT = "https://archive-api.open-meteo.com/v1/era5"

    def fetch_daily(self, points: dict, start_date: str, end_date: str, cache_dir: str = "data/cache") -> pd.DataFrame:
        point_ids = list(points.keys())
        lats = ",".join(str(points[p]["lat"]) for p in point_ids)
        lons = ",".join(str(points[p]["lon"]) for p in point_ids)

        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "timezone": "Asia/Kolkata"
        }
        res = cached_get_json(self.ENDPOINT, params, cache_dir=cache_dir)
        if isinstance(res, dict):
            res = [res]

        records = []
        for p_idx, p_id in enumerate(point_ids):
            data = res[p_idx]
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                continue
            df_h = pd.DataFrame({
                "time": pd.to_datetime(times),
                "t": hourly.get("temperature_2m"),
                "p": hourly.get("precipitation"),
                "w": hourly.get("wind_speed_10m")
            })
            df_h["date"] = df_h["time"].dt.strftime("%Y-%m-%d")
            # Aggregate to IST calendar day (00:00 to 23:00 IST)
            daily = df_h.groupby("date").agg({
                "t": "max",
                "p": "sum",
                "w": "max"
            }).reset_index()

            lat, lon = points[p_id]["lat"], points[p_id]["lon"]
            for _, r in daily.iterrows():
                d_str = r["date"]
                if pd.notna(r["t"]):
                    records.append((d_str, p_id, lat, lon, "temperature_2m_max", float(r["t"])))
                if pd.notna(r["p"]):
                    records.append((d_str, p_id, lat, lon, "precipitation_sum", max(0.0, float(r["p"]))))
                if pd.notna(r["w"]):
                    records.append((d_str, p_id, lat, lon, "wind_speed_10m_max", max(0.0, float(r["w"]))))

        return pd.DataFrame(records, columns=["date", "point", "lat", "lon", "variable", "truth"])
