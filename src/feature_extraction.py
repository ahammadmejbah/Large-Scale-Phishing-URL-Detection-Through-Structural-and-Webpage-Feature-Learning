"""Feature extraction utilities that turn a raw URL (and optionally its fetched
webpage) into the numeric feature vector expected by the trained model.

The PhiUSIIL dataset combines two families of signals:
  1. Lexical / structural features derived purely from the URL string.
  2. Webpage features derived from the fetched HTML (title, forms, scripts...).

For an interactive single-URL check we can compute (1) exactly and (2) only if
the user opts in to fetching the live page. When the page isn't fetched, the
webpage features fall back to dataset-derived median values (see
`feature_defaults.json`), which is clearly surfaced in the UI as an
approximation.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from difflib import SequenceMatcher
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 6
MAX_CONTENT_BYTES = 3_000_000  # 3 MB safety cap
MAX_REDIRECTS = 5

SOCIAL_DOMAINS = (
    "facebook.com", "twitter.com", "x.com", "instagram.com", "linkedin.com",
    "youtube.com", "tiktok.com", "pinterest.com", "whatsapp.com", "telegram.org",
)
EMPTY_HREFS = ("", "#", "javascript:void(0)", "javascript:;")


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _registrable_domain_parts(host: str) -> tuple[str, str]:
    """Return (domain_without_tld, tld) using a simple heuristic split."""
    parts = host.split(".")
    if len(parts) < 2:
        return host, ""
    return ".".join(parts[:-1]), parts[-1]


# ---------------------------------------------------------------------------
# Lexical (URL-only) features
# ---------------------------------------------------------------------------

def lexical_features(raw_url: str, tld_lookup: dict, char_freq: dict, legit_domains: list) -> dict:
    url = _normalize_url(raw_url)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    is_ip = _is_ip(host)

    _, tld = ("", "") if is_ip else _registrable_domain_parts(host)
    sub_domain_count = 0 if is_ip else max(len(host.split(".")) - 2, 0)

    letters = sum(ch.isalpha() for ch in url)
    digits = sum(ch.isdigit() for ch in url)
    n_equals = url.count("=")
    n_qmark = url.count("?")
    n_amp = url.count("&")
    special_chars = sum(1 for ch in url if not ch.isalnum())
    other_special = max(special_chars - n_equals - n_qmark - n_amp, 0)

    obfuscated_chars = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    url_len = max(len(url), 1)

    continuations = sum(1 for i in range(1, len(url)) if url[i] == url[i - 1])
    char_continuation_rate = continuations / max(len(url) - 1, 1)

    default_char_prob = (sum(char_freq.values()) / len(char_freq)) if char_freq else 0.05
    url_char_prob = (
        sum(char_freq.get(ch, default_char_prob) for ch in url) / len(url)
        if url else default_char_prob
    )

    default_tld_prob = (sum(tld_lookup.values()) / len(tld_lookup)) if tld_lookup else 0.5
    tld_legit_prob = tld_lookup.get(tld, default_tld_prob)

    similarity_index = _url_similarity_index(host, legit_domains)

    return {
        "URLLength": len(url),
        "DomainLength": len(host),
        "IsDomainIP": int(is_ip),
        "URLSimilarityIndex": similarity_index,
        "CharContinuationRate": char_continuation_rate,
        "TLDLegitimateProb": tld_legit_prob,
        "URLCharProb": url_char_prob,
        "TLDLength": len(tld),
        "NoOfSubDomain": sub_domain_count,
        "HasObfuscation": int(obfuscated_chars > 0),
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": (obfuscated_chars * 3) / url_len,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / url_len,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / url_len,
        "NoOfEqualsInURL": n_equals,
        "NoOfQMarkInURL": n_qmark,
        "NoOfAmpersandInURL": n_amp,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": special_chars / url_len,
        "IsHTTPS": int(parsed.scheme == "https"),
    }, host, tld


def _url_similarity_index(host: str, legit_domains: list) -> float:
    if not host:
        return 0.0
    if host in legit_domains:
        return 100.0
    candidates = [d for d in legit_domains if d and abs(len(d) - len(host)) <= 3 and d[0] == host[0]]
    if not candidates:
        candidates = legit_domains[:2000]
    best = 0.0
    for candidate in candidates:
        ratio = SequenceMatcher(None, host, candidate).ratio()
        if ratio > best:
            best = ratio
        if best >= 0.999:
            break
    return round(best * 100, 2)


# ---------------------------------------------------------------------------
# Live webpage fetch (opt-in) with SSRF protections
# ---------------------------------------------------------------------------

class UnsafeURLError(Exception):
    pass


def _assert_public_host(host: str) -> None:
    """Resolve the host and reject anything pointing at private/internal networks."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host: {host}") from exc

    for info in infos:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        ):
            raise UnsafeURLError(f"Refusing to fetch internal/private address: {ip_str}")


def fetch_webpage_features(raw_url: str) -> dict:
    """Fetch the URL and derive webpage-based features. Raises UnsafeURLError /
    requests exceptions on failure; caller should catch and fall back to defaults."""
    url = _normalize_url(raw_url)
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http/https URLs are supported")

    host = parsed.hostname or ""
    _assert_public_host(host)

    session = requests.Session()
    session.max_redirects = MAX_REDIRECTS
    resp = session.get(
        url,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0 (PhishingURLDetector/1.0)"},
    )
    content = resp.raw.read(MAX_CONTENT_BYTES + 1, decode_content=True)
    if len(content) > MAX_CONTENT_BYTES:
        content = content[:MAX_CONTENT_BYTES]
    html = content.decode(resp.encoding or "utf-8", errors="ignore")

    final_host = (urlsplit(resp.url).hostname or "").lower()
    redirect_count = len(resp.history)
    self_redirects = sum(
        1 for r in resp.history if (urlsplit(r.url).hostname or "").lower() == final_host
    )

    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    has_title = int(bool(title))

    def _match_score(text: str, reference: str) -> float:
        if not text or not reference:
            return 0.0
        return round(SequenceMatcher(None, text.lower(), reference.lower()).ratio() * 100, 2)

    domain_title_match = _match_score(final_host, title)
    url_title_match = _match_score(url, title)

    has_favicon = int(bool(soup.find("link", rel=lambda v: v and "icon" in v.lower())))
    has_description = int(bool(soup.find("meta", attrs={"name": "description"})))
    is_responsive = int(bool(soup.find("meta", attrs={"name": "viewport"})))

    forms = soup.find_all("form")
    has_external_form = 0
    for form in forms:
        action = (form.get("action") or "").strip()
        if action.startswith("http"):
            action_host = (urlsplit(action).hostname or "").lower()
            if action_host and action_host != final_host:
                has_external_form = 1
                break

    links = soup.find_all("a", href=True)
    self_ref = empty_ref = external_ref = 0
    has_social = 0
    for a in links:
        href = a["href"].strip()
        if href.lower() in EMPTY_HREFS or href.startswith("javascript:"):
            empty_ref += 1
            continue
        href_host = (urlsplit(href).hostname or "").lower()
        if not href_host or href_host == final_host:
            self_ref += 1
        else:
            external_ref += 1
            if any(social in href_host for social in SOCIAL_DOMAINS):
                has_social = 1

    inputs = soup.find_all("input")
    has_hidden = int(any((i.get("type") or "").lower() == "hidden" for i in inputs))
    has_password = int(any((i.get("type") or "").lower() == "password" for i in inputs))
    has_submit = int(
        any((i.get("type") or "").lower() == "submit" for i in inputs)
        or bool(soup.find("button", attrs={"type": "submit"}))
    )

    page_text = soup.get_text(" ", strip=True).lower()
    has_bank = int("bank" in page_text)
    has_pay = int(bool(re.search(r"\bpay(ment|pal)?\b", page_text)))
    has_crypto = int(bool(re.search(r"crypto|bitcoin|ethereum|wallet", page_text)))
    has_copyright = int("©" in html or "copyright" in page_text)

    scripts = soup.find_all("script")
    no_of_popup = sum(1 for s in scripts if s.string and "window.open" in s.string)

    robots_present = 0
    try:
        robots_url = f"{parsed.scheme}://{host}/robots.txt"
        r = session.get(robots_url, timeout=REQUEST_TIMEOUT)
        robots_present = int(r.status_code == 200)
    except requests.RequestException:
        robots_present = 0

    lines = html.splitlines() or [""]

    return {
        "LineOfCode": len(lines),
        "LargestLineLength": max(len(line) for line in lines),
        "HasTitle": has_title,
        "DomainTitleMatchScore": domain_title_match,
        "URLTitleMatchScore": url_title_match,
        "HasFavicon": has_favicon,
        "Robots": robots_present,
        "IsResponsive": is_responsive,
        "NoOfURLRedirect": redirect_count,
        "NoOfSelfRedirect": self_redirects,
        "HasDescription": has_description,
        "NoOfPopup": no_of_popup,
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": has_external_form,
        "HasSocialNet": has_social,
        "HasSubmitButton": has_submit,
        "HasHiddenFields": has_hidden,
        "HasPasswordField": has_password,
        "Bank": has_bank,
        "Pay": has_pay,
        "Crypto": has_crypto,
        "HasCopyrightInfo": has_copyright,
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": len(soup.find_all("link", rel="stylesheet")) + len(soup.find_all("style")),
        "NoOfJS": len(scripts),
        "NoOfSelfRef": self_ref,
        "NoOfEmptyRef": empty_ref,
        "NoOfExternalRef": external_ref,
    }, title
