"""Point-in-time SEC ingestion. No network is used by panel construction.

Companyfacts is an *aggregate*, not a submission register. Strict builds require
the submissions register so an absent first filing cannot become a later filing.
Annual inputs use original 10-K filings; quarterly inputs use the preceding
original 10-K or 10-Q. Panel builders exclude amended filings and unavailable
comparable EPS pairs rather than backfilling them from subsequent filings.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "interest_expense": ("InterestExpense", "InterestAndDebtExpense"),
    "income_tax": ("IncomeTaxExpenseBenefit",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_basic": ("EarningsPerShareBasic",),
    "shares_basic": ("WeightedAverageNumberOfSharesOutstandingBasic",),
    "assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "receivables": ("AccountsReceivableNetCurrent",),
    "inventory": ("InventoryNet",),
    "ppe": ("PropertyPlantAndEquipmentNet",),
    "liabilities": ("Liabilities",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "equity": ("StockholdersEquity",),
    "retained_earnings": ("RetainedEarningsAccumulatedDeficit",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}
INSTANT_ITEMS = frozenset(
    (
        "assets",
        "current_assets",
        "cash",
        "receivables",
        "inventory",
        "ppe",
        "liabilities",
        "current_liabilities",
        "long_term_debt",
        "equity",
        "retained_earnings",
    )
)
DEFAULT_CORE = (
    "revenue",
    "net_income",
    "eps_basic",
    "shares_basic",
    "assets",
    "liabilities",
    "equity",
)
_RATE_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


class DataExclusion(ValueError):
    """A documented, auditable reason to reject an observation."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def download_cached(
    url, destination, user_agent, *, public_url=None, max_bytes=100_000_000, retries=3
):
    """Rate limited, bounded, checksum-verified HTTP cache; secrets are never logged.

    A caller sending authentication in a query MUST supply a redacted public_url.
    The process-wide limiter is two requests/second; coordinate separate workers.
    """
    global _LAST_REQUEST
    destination = Path(destination)
    receipt_path = destination.with_name(destination.name + ".provenance.json")
    public_url = public_url or url
    if destination.exists():
        if not receipt_path.exists():
            raise ValueError(f"Cache lacks provenance receipt: {destination}")
        receipt = json.loads(receipt_path.read_text())
        if receipt["sha256"] != sha256_file(destination) or receipt["url"] != public_url:
            raise ValueError("Cache checksum or source URL differs; use a new destination")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        try:
            with _RATE_LOCK:
                time.sleep(max(0, 0.5 - (time.monotonic() - _LAST_REQUEST)))
                _LAST_REQUEST = time.monotonic()
            request = Request(
                url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"}
            )
            with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Download exceeds configured size limit")
                    output.write(chunk)
                headers = {
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
            # Hash only after the file is closed, so buffered bytes are included.
            receipt = {
                "url": public_url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "bytes": total,
                "sha256": sha256_file(temporary),
                **headers,
            }
            temporary.replace(destination)
            _atomic_json(receipt_path, receipt)
            return destination
        except HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HTTP {error.code} retrieving {public_url}") from None
            retry_after = error.headers.get("Retry-After", "")
            delay = (
                min(60.0, float(retry_after))
                if retry_after.isdigit()
                else min(60, 2 ** (attempt + 1))
            )
            time.sleep(delay)
        except URLError:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Network failure retrieving {public_url}") from None
            time.sleep(2 ** (attempt + 1))
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    raise RuntimeError("Download failed")


def _validate_sec_user_agent(user_agent):
    if not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", user_agent or "") or len(user_agent.split()) < 2:
        raise ValueError("SEC User-Agent must identify your organization/project and contact email")


def _cik(cik):
    text = str(cik).removeprefix("CIK")
    if not text.isdigit() or not 0 < int(text) < 10**10:
        raise ValueError("CIK must be a positive integer of at most ten digits")
    return f"{int(text):010d}"


def fetch_sec_companyfacts(cik, destination, user_agent):
    _validate_sec_user_agent(user_agent)
    return download_cached(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik(cik)}.json", destination, user_agent
    )


def fetch_sec_submissions(cik, destination, user_agent, include_history=True):
    """Download the register and all referenced historical chunks into a bundle.

    The destination is a directory, containing the original files plus a combined
    `submissions.json`. Loading the combined file suffices for a strict build.
    """
    _validate_sec_user_agent(user_agent)
    directory = Path(destination)
    path = download_cached(
        f"https://data.sec.gov/submissions/CIK{_cik(cik)}.json",
        directory / f"CIK{_cik(cik)}.json",
        user_agent,
    )
    value = json.loads(path.read_text())
    value["filings"]["history"] = []
    if include_history:
        for entry in value["filings"].get("files", []):
            name = entry["name"]
            if not re.fullmatch(r"CIK\d{10}-submissions-\d+\.json", name):
                raise ValueError("Unexpected SEC submission chunk filename")
            chunk = download_cached(
                f"https://data.sec.gov/submissions/{name}", directory / name, user_agent
            )
            value["filings"]["history"].append(json.loads(chunk.read_text()))
    value["_history_complete"] = bool(include_history or not value["filings"].get("files"))
    combined = directory / "submissions.json"
    _atomic_json(combined, value)
    return combined


def iter_local_json(paths):
    """Yield (source, JSON, sha256), reading ZIP members without extracting them."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded = []
    for path in paths:
        path = Path(path)
        # Each submission bundle is stored in submissions/CIK##########/.
        # Sort recursively so its documented parent-directory input round-trips
        # without depending on filesystem enumeration order.
        expanded.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    for path in expanded:
        if path.name.endswith(".provenance.json"):
            continue
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                for entry in sorted(archive.infolist(), key=lambda entry: entry.filename):
                    member = PurePosixPath(entry.filename)
                    if member.is_absolute() or ".." in member.parts or "\\" in entry.filename:
                        raise ValueError(f"Unsafe ZIP member: {entry.filename}")
                    if entry.is_dir() or member.suffix.lower() != ".json":
                        continue
                    if entry.file_size > 100_000_000:
                        raise ValueError("ZIP JSON member exceeds 100 MB limit")
                    raw = archive.read(entry)
                    yield f"{path}!{entry.filename}", json.loads(
                        raw, parse_float=Decimal
                    ), hashlib.sha256(raw).hexdigest()
        else:
            raw = path.read_bytes()
            if len(raw) > 100_000_000:
                raise ValueError("JSON source exceeds 100 MB limit")
            yield str(path), json.loads(raw, parse_float=Decimal), hashlib.sha256(raw).hexdigest()


def _columnar_rows(value):
    if isinstance(value, list):
        yield from value
        return
    required = ("accessionNumber", "filingDate", "reportDate", "form")
    if not all(key in value for key in required):
        return
    length = len(value["accessionNumber"])
    if any(len(value[key]) != length for key in required):
        raise ValueError("Malformed submissions column lengths")
    for index in range(length):
        yield {key: value[key][index] for key in required}


def load_submissions(paths):
    """Resolve native bulk main/chunk files or fetch_sec_submissions bundles."""
    if not paths:
        return {}
    loaded = list(iter_local_json(paths))
    objects = {Path(source.split("!")[-1]).name: value for source, value, _ in loaded}
    result = {}
    for source, value, digest in loaded:
        if "cik" not in value or "filings" not in value:
            continue
        rows = list(_columnar_rows(value["filings"].get("recent", {})))
        history = value["filings"].get("history", [])
        for chunk in history:
            rows.extend(_columnar_rows(chunk))
        complete = value.get("_history_complete", not value["filings"].get("files"))
        if not history and value["filings"].get("files"):
            complete = True
            for entry in value["filings"]["files"]:
                if entry["name"] not in objects:
                    complete = False
                else:
                    rows.extend(_columnar_rows(objects[entry["name"]]))
        index = {}
        for row in rows:
            if row["form"] != "10-K" or not row["reportDate"]:
                continue
            year = date.fromisoformat(row["reportDate"]).year
            previous = index.get(year)
            if previous is None or (row["filingDate"], row["accessionNumber"]) < (
                previous["filingDate"],
                previous["accessionNumber"],
            ):
                index[year] = row
        cik = _cik(value["cik"])
        previous = result.get(cik)
        if previous and previous["history_complete"] and not complete:
            continue
        if previous and previous["history_complete"] and complete and previous["years"] != index:
            raise ValueError(f"Conflicting complete submission registers for CIK {cik}")
        result[cik] = {
            "years": index,
            "history_complete": bool(complete),
            "source": source,
            "sha256": digest,
            "source_manifest": [{"source": s, "sha256": d} for s, _, d in loaded],
        }
    return result


def _decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise DataExclusion("non_numeric_fact") from None
    if not number.is_finite():
        raise DataExclusion("non_finite_fact")
    return number


def _annual(fact):
    try:
        days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days + 1
        return 350 <= days <= 378
    except (KeyError, ValueError, TypeError):
        return False


def _facts(company, tag, unit):
    return company.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit, [])


def _observed_filings(company):
    groups = defaultdict(list)
    for tag in {tag for tags in TAGS.values() for tag in tags}:
        units = company.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
        for facts in units.values():
            for fact in facts:
                if fact.get("form") == "10-K" and fact.get("fp") == "FY" and _annual(fact):
                    if fact["end"] < fact.get("filed", "") and fact.get("accn"):
                        groups[fact["accn"]].append(fact)
    index = {}
    for accession, facts in groups.items():
        end = max(fact["end"] for fact in facts)
        filed = min(fact["filed"] for fact in facts)
        year = date.fromisoformat(end).year
        row = {"accessionNumber": accession, "filingDate": filed, "reportDate": end, "form": "10-K"}
        old = index.get(year)
        if old is None or (filed, accession) < (old["filingDate"], old["accessionNumber"]):
            index[year] = row
    return index


def _pair(company, item, filing, prior_end=None, tags=None, instant=None):
    """Both years must use the same tag, accession, unit, and annual context."""
    unit = "shares" if item == "shares_basic" else "USD/shares" if item == "eps_basic" else "USD"
    accession, end = filing["accessionNumber"], filing["reportDate"]
    current_year = date.fromisoformat(end).year
    instant = item in INSTANT_ITEMS if instant is None else instant
    for tag in tags or TAGS[item]:
        rows = [
            fact
            for fact in _facts(company, tag, unit)
            if fact.get("accn") == accession
            and fact.get("form") == "10-K"
            and fact.get("filed") == filing["filingDate"]
            and (("start" not in fact) if instant else _annual(fact))
        ]
        chosen = []
        for year in (current_year - 1, current_year):
            target_end = end if year == current_year else prior_end
            options = [
                fact
                for fact in rows
                if (
                    fact["end"] == target_end
                    if target_end
                    else date.fromisoformat(fact["end"]).year == year
                    and date.fromisoformat(fact["end"]).month == 12
                )
            ]
            if not options:
                chosen = []
                break
            contexts = {(fact.get("start"), fact["end"], _decimal(fact["val"])) for fact in options}
            if len(contexts) > 1:
                raise DataExclusion(f"ambiguous_context:{item}")
            chosen.append(options[0])
        if chosen:
            return [str(_decimal(fact["val"])) for fact in chosen], {
                "tag": "us-gaap:" + tag,
                "unit": unit,
                "accession": accession,
                "filed": filing["filingDate"],
                "periods": [{"start": fact.get("start"), "end": fact["end"]} for fact in chosen],
                "reported_values": [str(fact["val"]) for fact in chosen],
            }
    raise DataExclusion(f"missing_pair:{item}")


def _derived_liabilities(company, filing, prior_end):
    """Total liabilities from the same original filing when the tag is absent:
    LiabilitiesAndStockholdersEquity minus equity including noncontrolling
    interest (else parent equity plus MinorityInterest). Opt-in; audited."""
    pair = lambda tag: _pair(company, "liabilities", filing, prior_end, tags=(tag,), instant=True)
    total, total_src = pair("LiabilitiesAndStockholdersEquity")
    minority = minority_src = None
    try:
        equity, equity_src = pair(
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        )
    except DataExclusion:
        equity, equity_src = pair("StockholdersEquity")
        try:
            minority, minority_src = pair("MinorityInterest")
        except DataExclusion:
            pass
    values = [
        str(_decimal(t) - _decimal(e) - (_decimal(minority[i]) if minority else 0))
        for i, (t, e) in enumerate(zip(total, equity))
    ]
    return values, {
        "tag": "derived:LiabilitiesAndStockholdersEquity-minus-equity",
        "unit": "USD",
        "accession": filing["accessionNumber"],
        "filed": filing["filingDate"],
        "periods": total_src["periods"],
        "derived_from": [total_src, equity_src] + ([minority_src] if minority_src else []),
        "reported_values": values,
    }


def build_sec_panel(
    paths,
    start_year=2012,
    end_year=2025,
    per_year=1000,
    seed=2026,
    *,
    submissions_paths=None,
    require_verified_first=True,
    eps_threshold=0.01,
    required_items=None,
    derive_liabilities=False,
):
    """Return (records, audit), with outcome years inclusively selected.

    Inputs may be companyfacts JSON files/directories or the official bulk ZIP.
    Strict mode requires complete submissions history for the first original 10-K.
    Optional absent facts stay absent; conflicting facts reject the observation.
    """
    if start_year > end_year or per_year is not None and per_year < 1 or eps_threshold < 0:
        raise ValueError("Invalid panel years, sample size, or EPS threshold")
    core = tuple(required_items or DEFAULT_CORE)
    if not set(core) <= TAGS.keys():
        raise ValueError("Unknown required item")
    submissions = load_submissions(submissions_paths)
    candidates, exclusions, sources = [], [], []
    seen = {}
    for source, company, digest in iter_local_json(paths):
        if "facts" not in company or "cik" not in company:
            exclusions.append({"source": source, "reason": "not_companyfacts"})
            continue
        cik = _cik(company["cik"])
        if cik in seen:
            if seen[cik] != digest:
                raise ValueError(
                    f"Different companyfacts snapshots for CIK {cik}; choose one immutable snapshot"
                )
            continue
        seen[cik] = digest
        sources.append({"source": source, "sha256": digest, "cik": cik})
        register = submissions.get(cik)
        verified = bool(register and register["history_complete"])
        if require_verified_first and not verified:
            exclusions.append({"firm_id": cik, "reason": "missing_complete_submissions_history"})
            continue
        filings = register["years"] if verified else _observed_filings(company)
        for outcome_year in range(start_year, end_year + 1):
            observation_id = f"SEC-{cik}-{outcome_year}"
            try:
                if outcome_year - 1 not in filings or outcome_year not in filings:
                    raise DataExclusion("missing_input_or_outcome_original_10k")
                input_filing, outcome_filing = filings[outcome_year - 1], filings[outcome_year]
                if any(
                    date.fromisoformat(filing["reportDate"]).month != 12
                    for filing in (input_filing, outcome_filing)
                ):
                    raise DataExclusion("not_december_fiscal_year_end")
                if input_filing["filingDate"] >= outcome_filing["filingDate"]:
                    raise DataExclusion("non_chronological_filings")
                if any(
                    filing["reportDate"] >= filing["filingDate"]
                    for filing in (input_filing, outcome_filing)
                ):
                    raise DataExclusion("filing_not_after_reporting_period")
                input_eps, eps_provenance = _pair(company, "eps_basic", input_filing)
                prior_end = eps_provenance["periods"][0]["end"]
                values, item_sources = {"eps_basic": input_eps}, {"eps_basic": eps_provenance}
                for item in TAGS:
                    if item == "eps_basic":
                        continue
                    try:
                        try:
                            values[item], item_sources[item] = _pair(
                                company, item, input_filing, prior_end
                            )
                        except DataExclusion as missing:
                            if not (
                                derive_liabilities
                                and item == "liabilities"
                                and str(missing).startswith("missing_pair:")
                            ):
                                raise
                            values[item], item_sources[item] = _derived_liabilities(
                                company, input_filing, prior_end
                            )
                        if (
                            item not in INSTANT_ITEMS
                            and item_sources[item]["periods"] != eps_provenance["periods"]
                        ):
                            raise DataExclusion(f"inconsistent_item_periods:{item}")
                    except DataExclusion as error:
                        if item in core or not str(error).startswith("missing_pair:"):
                            raise
                if "gross_profit" not in values and {"revenue", "cost_of_revenue"} <= values.keys():
                    values["gross_profit"] = [
                        str(_decimal(r) - _decimal(c))
                        for r, c in zip(values["revenue"], values["cost_of_revenue"])
                    ]
                    item_sources["gross_profit"] = {
                        "derived": "revenue - cost_of_revenue",
                        "accession": input_filing["accessionNumber"],
                    }
                if any(_decimal(value) <= 0 for value in values["assets"]):
                    raise DataExclusion("nonpositive_assets")
                if any(_decimal(value) <= 0 for value in values["shares_basic"]):
                    raise DataExclusion("nonpositive_shares")
                _, outcome_eps_source = _pair(
                    company, "eps_basic", outcome_filing, input_filing["reportDate"]
                )
                eps_prior, eps_current = map(_decimal, outcome_eps_source["reported_values"])
                change = eps_current - eps_prior
                if abs(change) < Decimal(str(eps_threshold)) or change == 0:
                    raise DataExclusion("eps_change_below_threshold")
                record = {
                    "id": observation_id,
                    "firm_id": cik,
                    "input_year": outcome_year - 1,
                    "outcome_year": outcome_year,
                    "years": [outcome_year - 2, outcome_year - 1],
                    "input_filed": input_filing["filingDate"],
                    "outcome_filed": outcome_filing["filingDate"],
                    "outcome_period_start": outcome_eps_source["periods"][1]["start"],
                    "currency": "USD",
                    "values": values,
                    "label": int(change > 0),
                    "synthetic": bool(company.get("synthetic", False)),
                    "provenance": {
                        "source": source,
                        "sha256": digest,
                        "input_accession": input_filing["accessionNumber"],
                        "outcome_accession": outcome_filing["accessionNumber"],
                        "items": item_sources,
                        "label_eps": outcome_eps_source,
                        "eps_change": str(change),
                        "first_original_verified": verified,
                        "submissions_source": (
                            {key: register[key] for key in ("source", "sha256")}
                            if verified
                            else None
                        ),
                        "accounting_residual_assets_minus_liabilities_equity": [
                            str(_decimal(a) - _decimal(l) - _decimal(e))
                            for a, l, e in zip(
                                values["assets"], values["liabilities"], values["equity"]
                            )
                        ],
                        "availability_definition": "full outcome 10-K filing date; earlier earnings releases are not tracked",
                    },
                }
                candidates.append(record)
            except (DataExclusion, KeyError, ValueError, TypeError) as error:
                exclusions.append(
                    {
                        "id": observation_id,
                        "firm_id": cik,
                        "outcome_year": outcome_year,
                        "reason": (
                            str(error)
                            if isinstance(error, DataExclusion)
                            else f"malformed_fact:{type(error).__name__}:{error}"
                        ),
                    }
                )
    selected = []
    sample_counts = {}
    for year in range(start_year, end_year + 1):
        rows = sorted(
            (row for row in candidates if row["outcome_year"] == year), key=lambda row: row["id"]
        )
        sample_counts[str(year)] = {
            "eligible": len(rows),
            "selected": min(len(rows), per_year) if per_year else len(rows),
        }
        if per_year and len(rows) > per_year:
            chosen_ids = {row["id"] for row in random.Random(seed + year).sample(rows, per_year)}
            for row in rows:
                if row["id"] not in chosen_ids:
                    exclusions.append(
                        {"id": row["id"], "reason": "not_selected_by_seeded_year_sample"}
                    )
            rows = [row for row in rows if row["id"] in chosen_ids]
        selected.extend(rows)
    audit = {
        "schema_version": 1,
        "source": "SEC companyfacts and submissions",
        "sources": sources,
        "submissions_sources": (
            next(iter(submissions.values()))["source_manifest"] if submissions else []
        ),
        "first_original_required": require_verified_first,
        "required_items": list(core),
        "eps_threshold": eps_threshold,
        "seed": seed,
        "sample_by_year": sample_counts,
        "records": len(selected),
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
        "exclusions": exclusions,
        "limitations": [
            "Companyfacts omits custom tags and detailed XBRL context IDs; conflicting entity facts are rejected.",
            "Full-filing availability is later than some earnings announcements; use an independently audited earliest-public-outcome date for exposure sensitivity.",
            "The SEC aggregate and current submissions register are snapshots, not historical archive versions; all input hashes must be retained.",
        ],
    }
    return selected, audit
