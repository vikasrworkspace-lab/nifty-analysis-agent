from flask import Flask, Response, abort
from google.cloud import storage
import os

app = Flask(__name__)

BUCKET = os.environ["GCS_CACHE_BUCKET"]
PREFIX = "dashboard"

ALLOWED_FILES = {
    "dashboard_data.json",
    "dashboard_data_banknifty.json",
    "dashboard_data_reliance.json",
}

client = storage.Client()
bucket = client.bucket(BUCKET)


@app.route("/<filename>")
def get_dashboard_data(filename):
    if filename not in ALLOWED_FILES:
        abort(404)

    blob = bucket.blob(f"{PREFIX}/{filename}")

    if not blob.exists():
        abort(404)

    data = blob.download_as_bytes()

    return Response(
        data,
        content_type="application/json",
        headers={
            "Cache-Control": "no-store, max-age=0"
        },
    )


@app.route("/health")
def health():
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
