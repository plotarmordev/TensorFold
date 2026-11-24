"""strata-job.sh's [kv] collection must print a reported dtype and fail when /metrics has none."""
import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

KV = Path(__file__).with_name("metrics-kv.py")


class Metrics(BaseHTTPRequestHandler):
    body = b"{}"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


def kv_dtype(payload):
    Metrics.body = json.dumps(payload).encode()
    server = HTTPServer(("127.0.0.1", 0), Metrics)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d/metrics" % server.server_port
        return subprocess.run([sys.executable, str(KV), url], capture_output=True, text=True)
    finally:
        server.shutdown()
        server.server_close()


class MetricsKvTests(unittest.TestCase):
    def test_prints_reported_dtype(self):
        result = kv_dtype({"engine": {"kv_dtype": "fp8"}})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "fp8")

    def test_fails_when_dtype_absent(self):
        for payload in ({"engine": {}}, {}, {"other": {"kv_dtype": "fp8"}}):
            result = kv_dtype(payload)
            self.assertNotEqual(result.returncode, 0, payload)

    def test_fails_when_dtype_not_a_string(self):
        for payload in ({"engine": {"kv_dtype": None}}, {"engine": {"kv_dtype": 8}}):
            result = kv_dtype(payload)
            self.assertNotEqual(result.returncode, 0, payload)


if __name__ == "__main__":
    unittest.main()
