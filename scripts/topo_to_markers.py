#!/usr/bin/env python3
"""Linear-reference topo axes and emit geographic markers for signals.

Station vertices (6 fields) are official chainage anchors. Intermediate
polyline vertices receive chainage proportional to geodesic length between
consecutive anchors. Signals are placed by interpolating that chainage
along the polyline.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

EARTH_R = 6371000.0  # mean Earth radius, metres


@dataclass
class Vertex:
    lat: float
    lon: float
    name: str | None = None
    code: str | None = None
    official_m: float | None = None
    internal: str | None = None
    chainage_m: float = 0.0
    geodesic_m: float = 0.0


@dataclass
class Signal:
    ident: str
    chainage_m: float
    track: int
    lat: float = 0.0
    lon: float = 0.0
    heading_deg: float = 0.0


@dataclass
class Axis:
    ident: str
    name: str
    vertices: list[Vertex] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = p2 - p1
    dlon = l2 - l1
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlon = l2 - l1
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_axes(text: str) -> list[Axis]:
    axes: list[Axis] = []
    current: Axis | None = None
    section = None  # poly | signal | other
    ident_re = re.compile(r"id\s*=\s*(\S+)")
    name_re = re.compile(r"name\s*=\s*(.+)")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "end topo":
            break
        if line == "axis":
            current = Axis(ident="", name="")
            section = None
            continue
        if line == "end axis":
            if current is not None:
                axes.append(current)
            current = None
            section = None
            continue
        if current is None:
            continue
        if line == "poly":
            section = "poly"
            continue
        if line == "end poly":
            section = None
            continue
        if line == "signal":
            section = "signal"
            continue
        if line == "end signal":
            section = None
            continue
        if section is None:
            m = ident_re.match(line)
            if m:
                current.ident = m.group(1).strip()
                continue
            m = name_re.match(line)
            if m:
                current.name = m.group(1).strip()
                continue
            continue
        if section == "poly":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
            except ValueError:
                continue
            vtx = Vertex(lat=lat, lon=lon)
            if len(parts) >= 6:
                vtx.name = parts[2]
                vtx.code = parts[3]
                vtx.official_m = float(parts[4])
                vtx.internal = parts[5]
            current.vertices.append(vtx)
        elif section == "signal":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            current.signals.append(
                Signal(ident=parts[0], chainage_m=float(parts[1]), track=int(float(parts[2])))
            )
    return axes


def assign_chainage(vertices: list[Vertex]) -> None:
    if not vertices:
        return
    vertices[0].geodesic_m = 0.0
    for i in range(1, len(vertices)):
        vertices[i].geodesic_m = vertices[i - 1].geodesic_m + haversine_m(
            vertices[i - 1].lat, vertices[i - 1].lon, vertices[i].lat, vertices[i].lon
        )

    anchors = [i for i, v in enumerate(vertices) if v.official_m is not None]
    if not anchors:
        for v in vertices:
            v.chainage_m = v.geodesic_m
        return

    # Vertices before the first official anchor inherit geodesic offset from it.
    first = anchors[0]
    for i in range(first + 1):
        vertices[i].chainage_m = (vertices[first].official_m or 0.0) + (
            vertices[i].geodesic_m - vertices[first].geodesic_m
        )

    for a, b in zip(anchors, anchors[1:]):
        c0 = vertices[a].official_m or 0.0
        c1 = vertices[b].official_m or 0.0
        s0 = vertices[a].geodesic_m
        s1 = vertices[b].geodesic_m
        span = s1 - s0
        for i in range(a, b + 1):
            t = 0.0 if span == 0 else (vertices[i].geodesic_m - s0) / span
            vertices[i].chainage_m = c0 + t * (c1 - c0)

    last = anchors[-1]
    for i in range(last, len(vertices)):
        vertices[i].chainage_m = (vertices[last].official_m or 0.0) + (
            vertices[i].geodesic_m - vertices[last].geodesic_m
        )


def locate(vertices: list[Vertex], chainage_m: float) -> tuple[float, float, float]:
    if not vertices:
        raise ValueError("empty polyline")
    if chainage_m <= vertices[0].chainage_m:
        lat, lon = vertices[0].lat, vertices[0].lon
        nxt = vertices[1] if len(vertices) > 1 else vertices[0]
        return lat, lon, initial_bearing_deg(lat, lon, nxt.lat, nxt.lon)
    if chainage_m >= vertices[-1].chainage_m:
        prev = vertices[-2] if len(vertices) > 1 else vertices[-1]
        lat, lon = vertices[-1].lat, vertices[-1].lon
        return lat, lon, initial_bearing_deg(prev.lat, prev.lon, lat, lon)

    for i in range(len(vertices) - 1):
        c0 = vertices[i].chainage_m
        c1 = vertices[i + 1].chainage_m
        lo, hi = (c0, c1) if c0 <= c1 else (c1, c0)
        if lo <= chainage_m <= hi or (c0 <= chainage_m <= c1):
            span = c1 - c0
            t = 0.0 if span == 0 else (chainage_m - c0) / span
            t = min(1.0, max(0.0, t))
            lat = vertices[i].lat + t * (vertices[i + 1].lat - vertices[i].lat)
            lon = vertices[i].lon + t * (vertices[i + 1].lon - vertices[i].lon)
            heading = initial_bearing_deg(
                vertices[i].lat, vertices[i].lon, vertices[i + 1].lat, vertices[i + 1].lon
            )
            return lat, lon, heading
    lat, lon = vertices[-1].lat, vertices[-1].lon
    return lat, lon, 0.0


def place_signals(axis: Axis) -> None:
    assign_chainage(axis.vertices)
    for sig in axis.signals:
        sig.lat, sig.lon, sig.heading_deg = locate(axis.vertices, sig.chainage_m)


def pk_label(metres: float) -> str:
    sign = "-" if metres < 0 else ""
    metres = abs(metres)
    km = int(metres // 1000)
    rem = metres - km * 1000
    return f"{sign}PK {km}+{rem:05.1f}"


def signal_kind(ident: str) -> str:
    prefix = ident.split("-", 1)[0].upper()
    if ident[:1].isdigit():
        return "Hito / otra"
    return {
        "A": "Avanzada",
        "E": "Entrada",
        "S": "Salida",
        "R": "Retroceso",
    }.get(prefix, prefix)


def signal_label(sig: Signal) -> str:
    return f"{sig.ident}  v{sig.track}  {pk_label(sig.chainage_m)}  {signal_kind(sig.ident)}"


def station_label(vertex: Vertex) -> str:
    return f"{vertex.name}  {pk_label(vertex.official_m or 0.0)}"


def mkr_quoted(name: str) -> str:
    return '"' + name.replace("\\", "\\\\").replace('"', "'") + '"'


def signal_color_abgr(ident: str) -> str:
    kind = signal_kind(ident)
    return {
        "Avanzada": "ff00a5ff",  # orange
        "Entrada": "ff0000ff",  # red
        "Salida": "ff00c800",  # green
        "Retroceso": "ffc000c0",  # magenta
    }.get(kind, "ff0080ff")  # blue


def write_kml(axis: Axis, path: Path) -> None:
    styles = {
        "Avanzada": ("ff00a5ff", "http://maps.google.com/mapfiles/kml/paddle/orange-blank.png"),
        "Entrada": ("ff0000ff", "http://maps.google.com/mapfiles/kml/paddle/red-blank.png"),
        "Salida": ("ff00c800", "http://maps.google.com/mapfiles/kml/paddle/grn-blank.png"),
        "Retroceso": ("ffc000c0", "http://maps.google.com/mapfiles/kml/paddle/purple-blank.png"),
        "Hito / otra": ("ff0080ff", "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png"),
        "Estacion": ("ffd18802", "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"),
    }
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "  <Document>",
        f"    <name>Señales {escape(axis.ident)} — {escape(axis.name)}</name>",
        "    <description>Marcadores interpolados sobre la polilínea del eje "
        "usando el PK oficial de las estaciones como anclas.</description>",
    ]
    for key, (color, href) in styles.items():
        style_id = re.sub(r"[^A-Za-z0-9]+", "", key)
        parts.append(
            f"""    <Style id="st-{style_id}">
      <IconStyle>
        <color>{color}</color>
        <scale>1.0</scale>
        <Icon><href>{href}</href></Icon>
        <hotSpot x="32" y="1" xunits="pixels" yunits="pixels"/>
      </IconStyle>
      <LabelStyle><scale>0.8</scale></LabelStyle>
      <LineStyle><color>ffaa00aa</color><width>3</width></LineStyle>
    </Style>"""
        )
    coords = " ".join(f"{v.lon:.10f},{v.lat:.10f},0" for v in axis.vertices)
    parts.append(
        f"""    <Placemark>
      <name>Eje {escape(axis.ident)}</name>
      <styleUrl>#st-Estacion</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coords}</coordinates>
      </LineString>
    </Placemark>"""
    )

    parts.append("    <Folder><name>Estaciones / puntos singulares</name>")
    for v in axis.vertices:
        if v.official_m is None or not v.name:
            continue
        parts.append(
            f"""      <Placemark>
        <name>{escape(station_label(v))}</name>
        <description>{escape(v.name)} ({escape(v.code or "")}) · {pk_label(v.official_m)}</description>
        <styleUrl>#st-Estacion</styleUrl>
        <Point><coordinates>{v.lon:.10f},{v.lat:.10f},0</coordinates></Point>
      </Placemark>"""
        )
    parts.append("    </Folder>")

    parts.append(f"    <Folder><name>Señales {escape(axis.ident)}</name>")
    for sig in sorted(axis.signals, key=lambda s: (s.chainage_m, s.ident)):
        kind = signal_kind(sig.ident)
        style_id = re.sub(r"[^A-Za-z0-9]+", "", kind)
        desc = (
            f"{escape(sig.ident)} · {pk_label(sig.chainage_m)} · "
            f"vía {sig.track} · {escape(kind)} · "
            f"{sig.lat:.6f}, {sig.lon:.6f}"
        )
        parts.append(
            f"""      <Placemark>
        <name>{escape(signal_label(sig))}</name>
        <description>{desc}</description>
        <styleUrl>#st-{style_id}</styleUrl>
        <Point><coordinates>{sig.lon:.10f},{sig.lat:.10f},0</coordinates></Point>
      </Placemark>"""
        )
    parts.append("    </Folder>")
    parts.append("  </Document>")
    parts.append("</kml>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_geojson(axis: Axis, path: Path) -> None:
    features = []
    features.append(
        {
            "type": "Feature",
            "properties": {"kind": "axis", "id": axis.ident, "name": axis.name},
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(v.lon, 8), round(v.lat, 8)] for v in axis.vertices],
            },
        }
    )
    for v in axis.vertices:
        if v.official_m is None or not v.name:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "station",
                    "name": v.name,
                    "code": v.code,
                    "chainage_m": v.official_m,
                    "pk": pk_label(v.official_m),
                },
                "geometry": {"type": "Point", "coordinates": [round(v.lon, 8), round(v.lat, 8)]},
            }
        )
    for sig in axis.signals:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "signal",
                    "id": sig.ident,
                    "chainage_m": sig.chainage_m,
                    "pk": pk_label(sig.chainage_m),
                    "track": sig.track,
                    "type": signal_kind(sig.ident),
                    "heading_deg": round(sig.heading_deg, 1),
                },
                "geometry": {"type": "Point", "coordinates": [round(sig.lon, 8), round(sig.lat, 8)]},
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_html(axis: Axis, path: Path) -> None:
    stations = [
        {
            "name": v.name,
            "code": v.code,
            "lat": v.lat,
            "lon": v.lon,
            "m": v.official_m,
            "pk": pk_label(v.official_m or 0),
        }
        for v in axis.vertices
        if v.official_m is not None and v.name
    ]
    signals = [
        {
            "id": s.ident,
            "lat": s.lat,
            "lon": s.lon,
            "m": s.chainage_m,
            "pk": pk_label(s.chainage_m),
            "track": s.track,
            "kind": signal_kind(s.ident),
            "heading": round(s.heading_deg, 1),
        }
        for s in sorted(axis.signals, key=lambda x: (x.chainage_m, x.ident))
    ]
    line = [[v.lat, v.lon] for v in axis.vertices]
    payload = json.dumps(
        {"axis": axis.ident, "name": axis.name, "line": line, "stations": stations, "signals": signals},
        ensure_ascii=False,
    )
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Señales {escape(axis.ident)} — {escape(axis.name)}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Segoe UI, sans-serif; }}
    #app {{ display: grid; grid-template-columns: 340px 1fr; height: 100%; }}
    #side {{ overflow: auto; border-right: 1px solid #ccc; background: #f7f7f7; }}
    #map {{ height: 100%; }}
    h1 {{ font-size: 16px; margin: 12px 16px 4px; }}
    p.meta {{ margin: 0 16px 12px; color: #444; font-size: 12px; }}
    .item {{ padding: 8px 16px; cursor: pointer; border-bottom: 1px solid #e5e5e5; font-size: 13px; }}
    .item:hover, .item.active {{ background: #fff3c4; }}
    .id {{ font-weight: 700; }}
    .pk {{ color: #333; }}
    .kind {{ color: #666; font-size: 11px; }}
    .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
    .legend {{ margin: 8px 16px; font-size: 12px; }}
    .legend span {{ margin-right: 10px; white-space: nowrap; }}
  </style>
</head>
<body>
  <div id="app">
    <aside id="side">
      <h1 id="title"></h1>
      <p class="meta" id="meta"></p>
      <div class="legend">
        <span><span class="dot" style="background:#ff8c00"></span>Avanzada</span>
        <span><span class="dot" style="background:#e00"></span>Entrada</span>
        <span><span class="dot" style="background:#0a0"></span>Salida</span>
        <span><span class="dot" style="background:#a0a"></span>Retroceso</span>
        <span><span class="dot" style="background:#08f"></span>Otra</span>
      </div>
      <div id="list"></div>
    </aside>
    <div id="map"></div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const data = {payload};
    const colors = {{
      "Avanzada": "#ff8c00",
      "Entrada": "#ee1111",
      "Salida": "#11aa22",
      "Retroceso": "#aa22aa",
      "Hito / otra": "#0088ff"
    }};
    document.getElementById("title").textContent = "Señales " + data.axis + " — " + data.name;
    document.getElementById("meta").textContent = data.signals.length + " señales interpoladas sobre la polilínea";

    const map = L.map("map");
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap"
    }}).addTo(map);
    const line = L.polyline(data.line, {{ color: "#aa00aa", weight: 4, opacity: 0.85 }}).addTo(map);
    map.fitBounds(line.getBounds(), {{ padding: [24, 24] }});

    data.stations.forEach(st => {{
      L.circleMarker([st.lat, st.lon], {{
        radius: 6, color: "#024e8c", weight: 2, fillColor: "#7ec8ff", fillOpacity: 1
      }}).addTo(map).bindPopup("<b>" + st.name + "</b><br>" + st.pk);
    }});

    const markers = [];
    const list = document.getElementById("list");
    data.signals.forEach((sg, i) => {{
      const color = colors[sg.kind] || "#0088ff";
      const m = L.circleMarker([sg.lat, sg.lon], {{
        radius: 7, color: "#222", weight: 1, fillColor: color, fillOpacity: 0.95
      }}).addTo(map);
      m.bindPopup(
        "<b>" + sg.id + "</b><br>" + sg.pk + "<br>Vía " + sg.track +
        "<br>" + sg.kind + "<br>" + sg.lat.toFixed(6) + ", " + sg.lon.toFixed(6)
      );
      markers.push(m);
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = '<span class="dot" style="background:' + color + '"></span>' +
        '<span class="id">' + sg.id + "</span> · <span class='pk'>" + sg.pk +
        "</span><div class='kind'>Vía " + sg.track + " · " + sg.kind + "</div>";
      div.onclick = () => {{
        document.querySelectorAll(".item").forEach(el => el.classList.remove("active"));
        div.classList.add("active");
        map.setView([sg.lat, sg.lon], 17);
        m.openPopup();
      }};
      list.appendChild(div);
    }});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_mkr(axis: Axis, path: Path) -> None:
    """MSTS marker file. TSRE5 lists it in the Navi / World Position window."""
    lines = ["SIMISA@@@@@@@@@@JINX0m1t______", ""]
    for v in axis.vertices:
        if v.official_m is None or not v.name:
            continue
        lines.append(
            f"Marker ( {v.lon:.8f} {v.lat:.8f} {mkr_quoted(station_label(v))} 0 )"
        )
    for sig in sorted(axis.signals, key=lambda s: (s.chainage_m, s.ident)):
        lines.append(
            f"Marker ( {sig.lon:.8f} {sig.lat:.8f} {mkr_quoted(signal_label(sig))} 0 )"
        )
    lines.append("")
    # UTF-16 LE with BOM, as classic MSTS SIMISA text.
    path.write_text("\n".join(lines), encoding="utf-16")


def write_gpx(axis: Axis, path: Path) -> None:
    """GPX waypoints + track. Alternative marker source for TSRE5 Navi."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="topo_to_markers"',
        '     xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <trk><name>Eje {escape(axis.ident)} {escape(axis.name)}</name><trkseg>",
    ]
    for v in axis.vertices:
        parts.append(f'    <trkpt lat="{v.lat:.8f}" lon="{v.lon:.8f}"></trkpt>')
    parts.append("  </trkseg></trk>")
    for v in axis.vertices:
        if v.official_m is None or not v.name:
            continue
        parts.append(
            f'  <wpt lat="{v.lat:.8f}" lon="{v.lon:.8f}">'
            f"<name>{escape(station_label(v))}</name></wpt>"
        )
    for sig in sorted(axis.signals, key=lambda s: (s.chainage_m, s.ident)):
        parts.append(
            f'  <wpt lat="{sig.lat:.8f}" lon="{sig.lon:.8f}">'
            f"<name>{escape(signal_label(sig))}</name></wpt>"
        )
    parts.append("</gpx>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_csv(axis: Axis, path: Path) -> None:
    lines = ["id,pk,chainage_m,track,kind,lat,lon,heading_deg"]
    for s in sorted(axis.signals, key=lambda x: (x.chainage_m, x.ident)):
        lines.append(
            f"{s.ident},{pk_label(s.chainage_m)},{s.chainage_m:.1f},{s.track},"
            f"{signal_kind(s.ident)},{s.lat:.8f},{s.lon:.8f},{s.heading_deg:.1f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def station_report(axis: Axis) -> Iterable[str]:
    yield f"Eje {axis.ident} ({axis.name}): {len(axis.vertices)} vértices, {len(axis.signals)} señales"
    for v in axis.vertices:
        if v.official_m is None:
            continue
        delta = v.geodesic_m - v.official_m
        yield (
            f"  {v.name:20s}  oficial {v.official_m:8.0f} m  "
            f"geodésico {v.geodesic_m:8.1f} m  Δ {delta:+7.1f} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Colocar señales de un eje topo sobre la polilínea.")
    parser.add_argument("topo", type=Path, help="Archivo topo (p.ej. toposfm227.raw)")
    parser.add_argument("--axis", default="T3", help="Identificador del eje (por defecto T3)")
    parser.add_argument("--out", type=Path, default=None, help="Directorio de salida (por defecto el del topo)")
    args = parser.parse_args()

    axes = parse_axes(read_text(args.topo))
    axis = next((a for a in axes if a.ident.upper() == args.axis.upper()), None)
    if axis is None:
        available = ", ".join(a.ident for a in axes) or "(ninguno)"
        raise SystemExit(f"Eje {args.axis!r} no encontrado. Disponibles: {available}")
    if not axis.vertices:
        raise SystemExit(f"El eje {axis.ident} no tiene polilínea.")
    if not axis.signals:
        raise SystemExit(f"El eje {axis.ident} no tiene señales.")

    place_signals(axis)
    out_dir = args.out or args.topo.parent
    stem = f"{axis.ident}_signals"
    kml = out_dir / f"{stem}.kml"
    html = out_dir / f"{stem}.html"
    geojson = out_dir / f"{stem}.geojson"
    csv = out_dir / f"{stem}.csv"
    mkr = out_dir / f"{stem}.mkr"
    gpx = out_dir / f"{stem}.gpx"
    write_kml(axis, kml)
    write_html(axis, html)
    write_geojson(axis, geojson)
    write_csv(axis, csv)
    write_mkr(axis, mkr)
    write_gpx(axis, gpx)

    print("\n".join(station_report(axis)))
    print()
    print(f"Escrito: {kml}")
    print(f"Escrito: {html}")
    print(f"Escrito: {geojson}")
    print(f"Escrito: {csv}")
    print(f"Escrito: {mkr}")
    print(f"Escrito: {gpx}")


if __name__ == "__main__":
    main()
