from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from fusion_strengthening_common import PROJECT, ROOT_OUT


RAW = PROJECT / "data" / "raw" / "faa_atcscc_advisories"
DEFAULT_EVENT_FILE = RAW / "faa_atcscc_gdp_gs_reparsed_2025_v2.csv"


def parse_months(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted({m for m in out if 1 <= m <= 12})


def detail_path(source_url: str) -> Path | None:
    parsed = urlparse(str(source_url))
    query = parse_qs(parsed.query)
    date_text = query.get("adv_date", [""])[0]
    advn = query.get("advn", [""])[0]
    if not date_text or not advn or len(date_text) != 8:
        return None
    month_dir = f"{date_text[4:8]}_{date_text[0:2]}"
    ymd = f"{date_text[4:8]}{date_text[0:2]}{date_text[2:4]}"
    return RAW / month_dir / f"detail_{ymd}_{int(advn)}.html"


def parse_signature_time(html: str) -> pd.Timestamp | pd.NaT:
    text = re.sub(r"&nbsp;?", " ", html)
    match = re.search(r"(\d{2})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2})", text)
    if not match:
        return pd.NaT
    yy, mm, dd, hh, minute = [int(x) for x in match.groups()]
    year = 2000 + yy
    return pd.Timestamp(year=year, month=mm, day=dd, hour=hh, minute=minute, tz="UTC")


def parse_adl_time(html: str, advisory_date: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"ADL TIME:\s*(\d{2})(\d{2})Z", html)
    if not match:
        return pd.NaT
    hh, minute = [int(x) for x in match.groups()]
    base = pd.Timestamp(advisory_date, tz="UTC")
    return base + pd.Timedelta(hours=hh, minutes=minute)


def effective_dt(advisory_date: str, day: int, hour: int, minute: int) -> pd.Timestamp:
    adv = pd.Timestamp(advisory_date, tz="UTC")
    candidate = pd.Timestamp(year=adv.year, month=adv.month, day=1, tz="UTC")
    candidate += pd.Timedelta(days=day - 1, hours=hour, minutes=minute)
    if candidate < adv - pd.Timedelta(days=15):
        candidate += pd.DateOffset(months=1)
    elif candidate > adv + pd.Timedelta(days=15):
        candidate -= pd.DateOffset(months=1)
    return candidate


def parse_effective_time(html: str, advisory_date: str) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    match = re.search(
        r"EFFECTIVE TIME:.*?<TD class=val>\s*(\d{2})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})",
        html,
        re.I | re.S,
    )
    if not match:
        return pd.NaT, pd.NaT
    sd, sh, sm, ed, eh, em = [int(x) for x in match.groups()]
    start = effective_dt(advisory_date, sd, sh, sm)
    end = effective_dt(advisory_date, ed, eh, em)
    if end < start:
        end += pd.Timedelta(days=1)
    return start, end


def parse_detail(row) -> dict:
    path = detail_path(row.source_url)
    out = {
        "source_url": row.source_url,
        "airport": row.airport,
        "tmi_type": row.tmi_type,
        "advisory_date": row.advisory_date,
        "detail_file": "" if path is None else str(path),
        "detail_exists": bool(path and path.exists()),
        "issue_utc": "",
        "adl_utc": "",
        "effective_start_utc": "",
        "effective_end_utc": "",
        "event_start_utc": row.start_utc,
        "issue_source": "missing",
        "lead_minutes": "",
        "event_start_delta_minutes": "",
    }
    if path is None or not path.exists():
        return out
    html = path.read_text(encoding="latin-1", errors="ignore")
    signature = parse_signature_time(html)
    adl = parse_adl_time(html, row.advisory_date)
    eff_start, eff_end = parse_effective_time(html, row.advisory_date)
    event_start = pd.Timestamp(row.start_utc)
    start = eff_start if pd.notna(eff_start) else event_start
    issue = signature if pd.notna(signature) else adl
    if pd.isna(issue):
        issue = start
        out["issue_source"] = "fallback_start"
    else:
        out["issue_source"] = "signature" if pd.notna(signature) else "adl_time"
    out["issue_utc"] = issue.isoformat()
    out["adl_utc"] = "" if pd.isna(adl) else adl.isoformat()
    out["effective_start_utc"] = "" if pd.isna(eff_start) else eff_start.isoformat()
    out["effective_end_utc"] = "" if pd.isna(eff_end) else eff_end.isoformat()
    out["lead_minutes"] = (start - issue).total_seconds() / 60.0
    out["event_start_delta_minutes"] = (start - event_start).total_seconds() / 60.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", default=str(DEFAULT_EVENT_FILE))
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--output-name", default="issue_time_smoke")
    args = parser.parse_args()

    months = parse_months(args.months)
    events = pd.read_csv(args.event_file)
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True)
    events = events[events["start_utc"].dt.month.isin(months)].copy()
    rows = [parse_detail(row) for row in events.itertuples(index=False)]
    detail = pd.DataFrame(rows)
    out_dir = ROOT_OUT / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "advisory_issue_times.csv", index=False)

    audit = detail.groupby("issue_source", as_index=False).agg(records=("source_url", "count"))
    audit["share"] = audit["records"] / len(detail) if len(detail) else 0
    audit.to_csv(out_dir / "advisory_issue_time_audit.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
