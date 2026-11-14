#!/usr/bin/env bash
# Download the OGLE-IV XROM photometry (phot.dat) of every source into one folder, named after the source.
#
#   ./get_ogle_xrom.sh                 # -> OGLE_XROM/sxp0.09.dat, OGLE_XROM/rx-j0520.5-6932.dat, ...
#   ./get_ogle_xrom.sh -o mydir        # other output folder
#   ./get_ogle_xrom.sh -f              # re-download files that already exist (default: skip them)
#   ./get_ogle_xrom.sh -m 'sxp*'       # only sources whose folder name matches a shell pattern
#
# Source: OGLE X-Ray Sources Monitor (XROM), https://ogle.astrouw.edu.pl/ogle4/xrom/xrom.html
#   data folders: https://www.astrouw.edu.pl/ogle/ogle4/xrom/<source>/phot.dat
#   columns: HJD, I [mag], sigma_I [mag], ...   (MJD = HJD - 2400000.5)
# Please acknowledge OGLE / XROM (Udalski et al. 2008, 2015) when you use the data.
#
# Works with curl (macOS, most Linux) or wget. Safe to re-run: existing files are kept unless -f is given.

set -u
BASE="${XROM_BASE:-https://www.astrouw.edu.pl/ogle/ogle4/xrom}"
OUT="OGLE_XROM"
FORCE=0
MATCH="*"
PAUSE=0.5                     # seconds between requests: be gentle with the OGLE server

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while getopts "o:fm:h" opt; do
    case $opt in
        o) OUT=$OPTARG ;;
        f) FORCE=1 ;;
        m) MATCH=$OPTARG ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

if command -v curl >/dev/null 2>&1; then
    fetch()    { curl -fsSL --retry 3 --connect-timeout 20 "$1"; }               # to stdout
    fetch_to() { curl -fsL --retry 3 --connect-timeout 20 -o "$2" "$1"; }       # to a file (quiet on 404)
elif command -v wget >/dev/null 2>&1; then
    fetch()    { wget -q --tries=3 --timeout=20 -O - "$1"; }
    fetch_to() { wget -q --tries=3 --timeout=20 -O "$2" "$1"; }
else
    echo "error: need curl or wget" >&2; exit 1
fi

mkdir -p "$OUT" || exit 1
echo "reading the source list from $BASE/ ..."
listing=$(fetch "$BASE/") || { echo "error: cannot read $BASE/" >&2; exit 1; }

# Folder links in the Apache index look like href="sxp0.09/". Keep only plain names ending in "/":
# skips the parent directory (href="/ogle/ogle4/"), the sort links (href="?C=N;O=D") and the files.
sources=$(printf '%s\n' "$listing" | grep -oE 'href="[^"/?][^"/]*/"' | sed -E 's/^href="(.*)\/"$/\1/' | sort -u)
[ -n "$sources" ] || { echo "error: no source folders found in the listing" >&2; exit 1; }
echo "$(printf '%s\n' "$sources" | wc -l | tr -d ' ') source folders listed"

n_new=0; n_skip=0; n_fail=0; failed=()
for src in $sources; do
    # shellcheck disable=SC2254
    case $src in $MATCH) ;; *) continue ;; esac
    dest="$OUT/$src.dat"
    if [ -s "$dest" ] && [ $FORCE -eq 0 ]; then
        n_skip=$((n_skip + 1)); continue
    fi
    tmp="$dest.part"
    if fetch_to "$BASE/$src/phot.dat" "$tmp" && [ -s "$tmp" ]; then
        mv "$tmp" "$dest"
        printf '  %-32s %6d points\n' "$src" "$(grep -cvE '^[[:space:]]*(#|$)' "$dest")"
        n_new=$((n_new + 1))
    else
        rm -f "$tmp"
        printf '  %-32s no phot.dat\n' "$src"
        failed+=("$src"); n_fail=$((n_fail + 1))
    fi
    sleep "$PAUSE"
done

echo "done: $n_new downloaded, $n_skip already present, $n_fail without phot.dat  ->  $OUT/"
[ "$MATCH" = "*" ] && rm -f "$OUT/missing.txt"
if [ $n_fail -gt 0 ]; then
    printf '%s\n' "${failed[@]}" > "$OUT/missing.txt"
    echo "folders without phot.dat are listed in $OUT/missing.txt (check them in a browser: $BASE/<name>/)"
fi
