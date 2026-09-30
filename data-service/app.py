from flask import Flask, Response, abort
from google.cloud import storage
import os

app = Flask(__name__)

BUCKET = os.environ["GCS_CACHE_BUCKET"]
PREFIX = "dashboard"

# Explicit allow-list, expressed as prefixes rather than an enumerated set of
# filenames so a newly added timeframe (intraday_5_, intraday_15_, ...) is
# publishable without touching this service. This is still an allow-list and
# deliberately NOT a general GCS proxy: only names matching one of these
# prefixes can ever be read, and the route is a single path segment.
#
# The dashboard group contributes two prefixes because its two shapes differ
# either side of the base name: "dashboard_data.json" and
# "dashboard_data_banknifty.json". Spelling both out stops a name like
# "dashboard_dataevil.json" from matching by accident, while still leaving
# every real sibling symbol servable.
ALLOWED_PREFIXES = ("dashboard_data.", "dashboard_data_", "intraday_")

client = storage.Client()
bucket = client.bucket(BUCKET)


def is_allowed(filename: str) -> bool:
    """Is this a dashboard artifact we are willing to serve?

    Three independent rejections, because the route is the only thing standing
    between the public internet and an entire bucket:
      * must be a ``.json`` artifact, never a blob of another type;
      * must start with an allowed prefix;
      * must not contain a path separator, so no traversal is expressible even
        if the route were ever widened to a path converter.
    """
    if not filename.endswith(".json"):
        return False
    if not filename.startswith(ALLOWED_PREFIXES):
        return False
    if "/" in filename or "\\" in filename or filename.startswith("."):
        return False
    return True


@app.route("/<filename>")
def get_dashboard_data(filename):
    if not is_allowed(filename):
        abort(404)

    blob = bucket.blob(f"{PREFIX}/{filename}")

    if not blob.exists():
        abort(404)

    # Reload so the metadata reflects the bytes we are actually returning; a
    # long-lived process would otherwise keep serving a stale generation.
    blob.reload()
    data = blob.download_as_bytes()

    return Response(
        data,
        content_type="application/json",
        headers={
            "Cache-Control": "no-store, max-age=0",
            # Diagnostic only. no-store already forbids reuse, but echoing the
            # object generation and update time makes a stale publish visible
            # from the response headers alone.
            "X-Served-Object-Generation": str(blob.generation),
            "X-Served-Object-Updated": str(blob.updated),
        },
    )


@app.route("/health")
def health():
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
