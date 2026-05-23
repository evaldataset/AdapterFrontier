#!/bin/bash
# Pre-submission sanity check. Run on D-1 before uploading to OpenReview.
# Exits non-zero on any failure.

set -uo pipefail
cd "$(dirname "$0")/.."

TEX="paper/emnlp.tex"
PDF="paper/emnlp.pdf"
FAIL=0

echo "=== ACL/EMNLP Submission Verification ==="
echo "$(date '+%F %T')"
echo

# 1. Anonymity — no author info leaked
echo "[1] Anonymity check"
LEAK=$(grep -c -iE '(suan|@gmail|@suanlab|oem-System-Product-Name|/home/suan|/projects/Adapter)' "$TEX" || true)
if [ "$LEAK" -gt 0 ]; then
    echo "  FAIL: $LEAK suspicious author-info string(s) in source"
    grep -niE '(suan|@gmail|@suanlab|oem-System-Product-Name|/home/suan|/projects/Adapter)' "$TEX" | head -5
    FAIL=$((FAIL+1))
else
    echo "  OK"
fi

# 2. Acknowledgments must NOT appear in review version
echo "[2] Acknowledgments forbidden in [review] mode"
ACK=$(grep -cE '\\section\*?\{(Acknowledg|acknowledg)' "$TEX" || true)
if [ "$ACK" -gt 0 ]; then
    echo "  FAIL: Acknowledgments section present"
    FAIL=$((FAIL+1))
else
    echo "  OK"
fi

# 3. Limitations section must be present and unnumbered
echo "[3] Limitations section (unnumbered, mandatory)"
LIM=$(grep -c '\\section\*{Limitations}' "$TEX" || true)
if [ "$LIM" -eq 0 ]; then
    echo "  FAIL: missing \\section*{Limitations}"
    FAIL=$((FAIL+1))
else
    echo "  OK"
fi

# 4. ACL [review] mode active
echo "[4] ACL [review] mode active"
REV=$(grep -c '\\usepackage\[review\]{acl}' "$TEX" || true)
if [ "$REV" -eq 0 ]; then
    echo "  FAIL: \\usepackage[review]{acl} not found"
    FAIL=$((FAIL+1))
else
    echo "  OK"
fi

# 5. Style files unmodified vs official
echo "[5] acl.sty + acl_natbib.bst byte-identical to official"
for f in acl.sty acl_natbib.bst; do
    OFFICIAL=$(gh api repos/acl-org/acl-style-files/contents/$f --jq .content 2>/dev/null | base64 -d 2>/dev/null | md5sum | awk '{print $1}')
    LOCAL=$(md5sum "paper/$f" 2>/dev/null | awk '{print $1}')
    if [ -z "$OFFICIAL" ] || [ -z "$LOCAL" ] || [ "$OFFICIAL" != "$LOCAL" ]; then
        echo "  FAIL: $f differs from official (or fetch failed)"
        FAIL=$((FAIL+1))
    else
        echo "  OK $f"
    fi
done

# 6. PDF compiled, no LaTeX errors
echo "[6] PDF compiles + no LaTeX errors"
if [ ! -f "$PDF" ]; then
    echo "  FAIL: PDF not present"
    FAIL=$((FAIL+1))
else
    PAGES=$(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages/{print $2}')
    SIZE=$(pdfinfo "$PDF" 2>/dev/null | awk -F: '/^Page size/{print $2}' | xargs)
    echo "  pages: $PAGES   size: $SIZE"
    if [ -z "$PAGES" ]; then
        echo "  FAIL: pdfinfo failed"
        FAIL=$((FAIL+1))
    fi
fi

# 7. A4 paper size (ACL requirement)
echo "[7] A4 paper size (ACL requires)"
A4=$(pdfinfo "$PDF" 2>/dev/null | grep -c '595.276 x 841.89')
if [ "$A4" -eq 0 ]; then
    echo "  FAIL: not A4 (must be 595.276 x 841.89 pts)"
    FAIL=$((FAIL+1))
else
    echo "  OK"
fi

# 8. Body ≤ 8 pages (Limitations starts on/before page 8)
echo "[8] Main body ≤ 8 pages"
if [ -f "$PDF" ]; then
    LIM_PAGE=""
    NPAGES=$(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages/{print $2}')
    NPAGES=${NPAGES:-12}
    for p in $(seq 1 $NPAGES); do
        # Avoid pipefail SIGPIPE issue: write to tmpfile, then grep
        TMP=$(mktemp)
        pdftotext -f $p -l $p "$PDF" "$TMP" 2>/dev/null
        if grep -E '^[[:space:]]*Limitations[[:space:]]*$' "$TMP" >/dev/null 2>&1; then
            LIM_PAGE=$p
            rm -f "$TMP"
            break
        fi
        rm -f "$TMP"
    done
    if [ -z "$LIM_PAGE" ]; then
        echo "  FAIL: Limitations heading not found in PDF"
        FAIL=$((FAIL+1))
    elif [ "$LIM_PAGE" -gt 9 ]; then
        echo "  FAIL: Limitations on page $LIM_PAGE (>9; body exceeds 8p)"
        FAIL=$((FAIL+1))
    else
        echo "  OK Limitations starts page $LIM_PAGE (body fills p1-$((LIM_PAGE-1)))"
    fi
fi

# 9. Anonymous URL is anonymous.4open.science (NOT dropbox/etc)
echo "[9] Anonymous URL points to anonymous.4open.science"
URL_BAD=$(grep -cE 'dropbox\.com|drive\.google|onedrive|/github\.com' "$TEX" | head -1)
URL_OK=$(grep -cE 'anonymous\.4open\.science' "$TEX")
if [ "$URL_BAD" -gt 0 ]; then
    echo "  FAIL: tracking-enabled URL host detected in tex"
    FAIL=$((FAIL+1))
elif [ "$URL_OK" -eq 0 ]; then
    echo "  WARN: no anonymous.4open.science URL found (may be intentional if no code release)"
else
    echo "  OK"
fi

# 10. URL hash placeholder XXXX should be replaced at D-1
echo "[10] Anonymous URL XXXX placeholder (D-1 must replace)"
XXXX=$(grep -c 'AdapterFrontier-XXXX' "$TEX")
if [ "$XXXX" -gt 0 ]; then
    echo "  TODO: replace 'XXXX' with real anonymous.4open.science slug"
fi

# 11. No prompt injection bait strings (auto-desk-reject trigger)
echo "[11] No prompt-injection bait strings"
INJECT=$(grep -ciE '(ignore (previous|all) instructions|system prompt|you are now|reviewer.*accept this paper)' "$TEX" || true)
if [ "$INJECT" -gt 0 ]; then
    echo "  FAIL: suspected prompt-injection string found"
    FAIL=$((FAIL+1))
else
    echo "  OK"
fi

# 12. All \ref{} resolve (no ??)
echo "[12] All references resolve (no ??)"
UNDEF=$(pdftotext "$PDF" - 2>/dev/null | grep -c '??' || true)
if [ "$UNDEF" -gt 5 ]; then  # >5 because '??' may appear legitimately
    echo "  WARN: $UNDEF '??' markers in PDF (may be unresolved refs)"
fi

# 13. Bibliography before \appendix
echo "[13] References before \\appendix (ARR rule)"
BIB_LINE=$(grep -n '\\begin{thebibliography}' "$TEX" | cut -d: -f1 | head -1)
APP_LINE=$(grep -n '^\\appendix' "$TEX" | cut -d: -f1 | head -1)
if [ -z "$BIB_LINE" ] || [ -z "$APP_LINE" ]; then
    echo "  WARN: could not locate bibliography or appendix"
elif [ "$BIB_LINE" -gt "$APP_LINE" ]; then
    echo "  FAIL: bibliography (line $BIB_LINE) after appendix (line $APP_LINE)"
    FAIL=$((FAIL+1))
else
    echo "  OK (bib L$BIB_LINE before appendix L$APP_LINE)"
fi

echo
echo "=== Summary: $FAIL failure(s) ==="
if [ "$FAIL" -eq 0 ]; then
    echo "OK — paper appears submission-ready."
    exit 0
else
    echo "DO NOT SUBMIT until all FAIL items are resolved."
    exit 1
fi
