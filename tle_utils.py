import requests
from sgp4.api import Satrec
from datetime import datetime
import pandas as pd

TLE_SOURCES = {
    "All Active Satellites": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    "Starlink":           "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "Iridium":            "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium&FORMAT=tle"
}


def fetch_tle(source_url):
    response = requests.get(source_url)
    lines = response.text.strip().splitlines()
    return lines

def compute_positions(tle_lines, max_sats=20):
    sats = []
    now = datetime.utcnow()

    for i in range(0, min(len(tle_lines), max_sats * 3), 3):
        # Ensure we have a complete TLE group
        if i + 2 >= len(tle_lines):
            break

        name = tle_lines[i].strip()
        try:
            s = Satrec.twoline2rv(tle_lines[i+1], tle_lines[i+2])

            error_code, pos, vel = s.sgp4(
                now.year,
                now.timetuple().tm_yday + now.hour / 24 + now.minute / (24 * 60)
            )

            if error_code == 0:
                sats.append({
                    'name': name,
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                    'vx': vel[0],
                    'vy': vel[1],
                    'vz': vel[2]
                })
        except Exception as e:
            # Skip malformed TLEs or sgp4 failures
            continue

    return pd.DataFrame(sats)
