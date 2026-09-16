#!/usr/bin/env python3
"""Bounded, resumable direct download; verify the publisher's wheel checksum."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import time
import urllib.parse
import urllib.request


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        if tag == "a":
            self.links.extend(value for name, value in attributes if name == "href")


def write_status(path, value):
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def hash_file(path, algorithm="sha256"):
    checksum = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(data)
    return checksum.hexdigest()


def run(arguments):
    destination = arguments.destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm = "sha256"
    if arguments.index:
        request = urllib.request.Request(arguments.index, headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=90) as response:
            text = response.read().decode()
        parser = Links()
        parser.feed(text)
        matches = [urllib.parse.urljoin(arguments.index, item) for item in parser.links
                   if urllib.parse.unquote(urllib.parse.urlsplit(item).path).endswith("/" + destination.name)]
        if len(matches) != 1:
            raise RuntimeError("wheel must have exactly one match on the official index")
        parsed = urllib.parse.urlsplit(matches[0])
        digest = urllib.parse.parse_qs(parsed.fragment).get("sha256", [None])[0]
        if digest is None or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("official wheel SHA256 missing")
        url = urllib.parse.urlunsplit(parsed._replace(fragment=""))
    else:
        url = arguments.url
        if arguments.expected_sha256:
            digest = arguments.expected_sha256
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError("invalid direct-download publisher SHA256")
        else:
            digest = arguments.expected_md5
            algorithm = "md5"
            if not digest or not re.fullmatch(r"[0-9a-f]{32}", digest):
                raise RuntimeError("direct CDN download requires a publisher checksum")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        length = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
    directory = destination.with_suffix(destination.suffix + ".parts")
    directory.mkdir(exist_ok=True)
    if algorithm == "md5" and etag.strip('"') != digest:
        raise RuntimeError("direct CDN ETag changed")
    contract = {"url": url, "checksum_algorithm": algorithm, "expected_digest": digest,
                "bytes": length, "etag": etag, "chunk_bytes": arguments.chunk_kib * 1024}
    binding = directory / "contract.json"
    if binding.exists() and json.loads(binding.read_text()) != contract:
        raise RuntimeError("download source changed; do not mix ranges")
    write_status(binding, contract)
    if destination.exists():
        if destination.stat().st_size != length or hash_file(destination, algorithm) != digest:
            raise RuntimeError("existing wheel is not the publisher-verified artifact")
        print("Existing wheel verified", destination.name, flush=True)
        return
    chunk_bytes = contract["chunk_bytes"]
    chunks = [(index, start, min(length, start + chunk_bytes) - 1)
              for index, start in enumerate(range(0, length, chunk_bytes))]

    def download_chunk(item):
        index, start, end = item
        path = directory / f"{index:04d}.part"
        expected = end - start + 1
        for attempt in range(5):
            downloaded = path.stat().st_size if path.exists() else 0
            if downloaded == expected:
                return
            if downloaded > expected:
                raise RuntimeError("oversized range fragment")
            position = start + downloaded
            headers = {"Range": f"bytes={position}-{end}", "Accept-Encoding": "identity"}
            if etag:
                headers["If-Range"] = etag
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {position}-{end}/{length}":
                        raise RuntimeError("server did not honor the precise range contract")
                    with path.open("ab") as handle:
                        while True:
                            data = response.read(256 * 1024)
                            if not data:
                                break
                            handle.write(data)
                if path.stat().st_size == expected:
                    return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 + attempt)
        raise RuntimeError("incomplete range after bounded retries")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        pending = {executor.submit(download_chunk, item) for item in chunks}
        while pending:
            finished, pending = wait(pending, timeout=10)
            for future in finished:
                future.result()
            received = sum(path.stat().st_size for path in directory.glob("*.part"))
            write_status(directory / "status.json", {"status": "DOWNLOADING", "received_bytes": received,
                         "total_bytes": length, "workers": arguments.workers,
                         "bytes_per_second": received / max(time.perf_counter() - started, 1),
                         "updated_at": datetime.now().astimezone().isoformat()})
    partial = destination.with_suffix(destination.suffix + ".assembled")
    with partial.open("wb") as handle:
        for index, _start, _end in chunks:
            with (directory / f"{index:04d}.part").open("rb") as fragment:
                shutil.copyfileobj(fragment, handle, 8 * 1024 * 1024)
    if partial.stat().st_size != length or hash_file(partial, algorithm) != digest:
        raise RuntimeError("assembled wheel failed publisher SHA256 verification")
    partial.replace(destination)
    write_status(directory / "status.json", {"status": "CDN_CHECKSUM_VERIFIED", **contract,
                 "recorded_sha256": hash_file(destination),
                 "seconds": time.perf_counter() - started, "finished_at": datetime.now().astimezone().isoformat()})
    print("CDN checksum verified", destination.name, algorithm, digest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index")
    source.add_argument("--url")
    parser.add_argument("--expected-md5")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--chunk-kib", type=int, default=16384)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("at most eight direct download streams")
    run(arguments)
