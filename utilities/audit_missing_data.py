"""
ATP Player Biographical Data Auditor
Scans the JeffSackmann/tennis_atp player database for missing or malformed
biographical data and generates a structured report for the open-source community.
"""

import csv
import os
import sys
import json
import datetime
import argparse
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PLAYER_FILE_PATH = os.path.join('..', 'atp_players.csv')
OUTPUT_CSV       = 'missing_bio_report.csv'
OUTPUT_JSON      = 'missing_bio_report.json'
OUTPUT_MD        = 'missing_bio_report.md'

# Known placeholder values that indicate missing data
MISSING_HAND_VALUES = {'', 'U', 'u', 'unknown'}
MISSING_DOB_VALUES  = {'', '00000000', '0', 'none'}
VALID_HAND_VALUES   = {'R', 'L', 'A'}  # Right, Left, Ambidextrous
VALID_COUNTRY_LEN   = 3  # IOC 3-letter country codes


# ─────────────────────────────────────────────
# AUDIT LOGIC
# ─────────────────────────────────────────────
def parse_dob(dob_str: str):
    """Attempt to parse DOB string YYYYMMDD. Returns datetime or None."""
    try:
        return datetime.datetime.strptime(dob_str.strip(), '%Y%m%d')
    except (ValueError, AttributeError):
        return None


def audit_player_row(row: list, line_num: int) -> dict:
    """
    Audits a single player row for data quality issues.
    ATP player file columns: ID, First, Last, Hand, DOB, Country
    Returns a structured result dict.
    """
    result = {
        'line':           line_num,
        'player_id':      None,
        'name':           None,
        'missing_fields': [],
        'warnings':       [],
        'critical':       [],
    }

    if len(row) < 5:
        result['critical'].append(f"Malformed row — only {len(row)} columns (expected 6)")
        return result

    p_id    = row[0].strip()
    first   = row[1].strip()
    last    = row[2].strip()
    hand    = row[3].strip()
    dob     = row[4].strip()
    country = row[5].strip() if len(row) > 5 else ''

    result['player_id'] = p_id
    result['name']      = f"{first} {last}".strip()

    # ── Missing field checks ──
    if not first and not last:
        result['critical'].append("No name data at all")
    elif not first:
        result['warnings'].append("Missing first name")
    elif not last:
        result['warnings'].append("Missing last name")

    if hand.upper() in MISSING_HAND_VALUES:
        result['missing_fields'].append('Hand')
    elif hand.upper() not in VALID_HAND_VALUES:
        result['warnings'].append(f"Unrecognized hand value: '{hand}'")

    if dob in MISSING_DOB_VALUES:
        result['missing_fields'].append('Birthdate')
    else:
        parsed_dob = parse_dob(dob)
        if parsed_dob is None:
            result['warnings'].append(f"Unparseable DOB format: '{dob}'")
        elif parsed_dob.year < 1880 or parsed_dob.year > datetime.datetime.now().year:
            result['warnings'].append(f"DOB year {parsed_dob.year} outside plausible range")

    if not country:
        result['missing_fields'].append('Country')
    elif len(country) != VALID_COUNTRY_LEN:
        result['warnings'].append(f"Country code '{country}' is not a 3-letter IOC code")

    if not p_id:
        result['critical'].append("Missing player ID — row cannot be cross-referenced")

    return result


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────
def write_csv_report(flagged: list, output_path: str):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Player ID', 'Name', 'Missing Fields', 'Warnings', 'Critical', 'Line #'])
        for r in flagged:
            writer.writerow([
                r['player_id'],
                r['name'],
                '; '.join(r['missing_fields']),
                '; '.join(r['warnings']),
                '; '.join(r['critical']),
                r['line'],
            ])


def write_json_report(summary: dict, flagged: list, output_path: str):
    report = {
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'summary':      summary,
        'flagged':      flagged,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def write_markdown_report(summary: dict, flagged: list, output_path: str):
    """Generates a GitHub-flavored markdown report for community visibility."""
    lines = [
        "# ATP Player Biographical Data Audit",
        f"*Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total Players Scanned | {summary['total_players']:,} |",
        f"| Players with Missing Data | {summary['total_flagged']:,} |",
        f"| Missing Birthdate | {summary['missing_dob']:,} |",
        f"| Missing Hand | {summary['missing_hand']:,} |",
        f"| Missing Country | {summary['missing_country']:,} |",
        f"| Critical Errors | {summary['critical_errors']:,} |",
        f"| Data Completeness | {summary['completeness_pct']:.1f}% |",
        "",
        "## Most Common Issues",
        "",
    ]

    for field, count in sorted(summary['missing_by_field'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- **{field}**: {count:,} players ({count/summary['total_players']*100:.1f}%)")

    lines += [
        "",
        "## Sample Flagged Records (first 20)",
        "",
        "| Player ID | Name | Missing Fields | Warnings |",
        "|-----------|------|----------------|----------|",
    ]

    for r in flagged[:20]:
        missing = ', '.join(r['missing_fields']) or '—'
        warnings = ', '.join(r['warnings']) or '—'
        lines.append(f"| {r['player_id']} | {r['name']} | {missing} | {warnings} |")

    lines += [
        "",
        "---",
        "*This report was generated automatically by the ATP biographical data auditor.*",
        "*Full data available in `missing_bio_report.csv` and `missing_bio_report.json`.*",
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ATP Player Biographical Data Auditor')
    parser.add_argument('--player-file', default=PLAYER_FILE_PATH, help='Path to atp_players.csv')
    parser.add_argument('--output-dir',  default='.',              help='Directory for output reports')
    parser.add_argument('--format',      default='all',            choices=['csv', 'json', 'md', 'all'])
    args = parser.parse_args()

    player_file = Path(args.player_file)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not player_file.exists():
        print(f"❌ Player file not found: {player_file}")
        print("Run this script from the 'utilities' directory, or pass --player-file explicitly.")
        sys.exit(1)

    print(f"🔍 Scanning {player_file}...")

    flagged      = []
    total        = 0
    missing_by_field = defaultdict(int)
    critical_count   = 0

    with open(player_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            total += 1
            result = audit_player_row(row, line_num)

            has_issue = result['missing_fields'] or result['warnings'] or result['critical']
            if has_issue:
                flagged.append(result)

            for field in result['missing_fields']:
                missing_by_field[field] += 1
            if result['critical']:
                critical_count += 1

    summary = {
        'total_players':    total,
        'total_flagged':    len(flagged),
        'missing_dob':      missing_by_field.get('Birthdate', 0),
        'missing_hand':     missing_by_field.get('Hand', 0),
        'missing_country':  missing_by_field.get('Country', 0),
        'critical_errors':  critical_count,
        'completeness_pct': ((total - len(flagged)) / total * 100) if total > 0 else 0,
        'missing_by_field': dict(missing_by_field),
    }

    # Write reports
    if args.format in ('csv', 'all'):
        path = output_dir / OUTPUT_CSV
        write_csv_report(flagged, str(path))
        print(f"📄 CSV report: {path}")

    if args.format in ('json', 'all'):
        path = output_dir / OUTPUT_JSON
        write_json_report(summary, flagged, str(path))
        print(f"📄 JSON report: {path}")

    if args.format in ('md', 'all'):
        path = output_dir / OUTPUT_MD
        write_markdown_report(summary, flagged, str(path))
        print(f"📄 Markdown report: {path}")

    # Console summary
    print("\n" + "═" * 55)
    print("  ATP BIOGRAPHICAL DATA AUDIT COMPLETE")
    print("═" * 55)
    print(f"  Total Players Scanned:  {total:,}")
    print(f"  Players Flagged:        {len(flagged):,}")
    print(f"  Missing Birthdate:      {summary['missing_dob']:,}")
    print(f"  Missing Hand:           {summary['missing_hand']:,}")
    print(f"  Missing Country:        {summary['missing_country']:,}")
    print(f"  Critical Errors:        {critical_count:,}")
    print(f"  Data Completeness:      {summary['completeness_pct']:.1f}%")
    print("═" * 55)

    if critical_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
