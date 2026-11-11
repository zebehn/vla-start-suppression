"""Stub adapter servers with injected faults must fail the matching conformance checks (needs pyzmq, msgpack_numpy)."""
import os
import socket
import subprocess
import sys
import time
import unittest

from tests._util import BENCH, have


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipUnless(have("zmq", "msgpack", "msgpack_numpy"), "pyzmq/msgpack_numpy not installed")
class Conformance(unittest.TestCase):
    def run_stub(self, *flags):
        from wnta import conformance as CF
        from wnta import protocol as P
        port = free_port()
        env = dict(os.environ, PYTHONPATH=BENCH)
        proc = subprocess.Popen([sys.executable, "-m", "wnta.adapters.base", "--port", str(port), *flags],
                                cwd=BENCH, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            cl = P.PolicyClient(port, timeout_ms=2000)
            for _ in range(50):
                if cl.ping():
                    break
                time.sleep(0.1)
            cl.close()
            rep = CF.run(port, timeout_ms=5000)
            P.PolicyClient(port, timeout_ms=2000).call("kill")
            return {r["id"]: r["status"] for r in rep}
        finally:
            proc.wait(timeout=10) if proc.poll() is None else None

    def test_clean_stub_passes(self):
        st = self.run_stub()
        self.assertFalse([k for k, v in st.items() if v == "fail"], st)

    def test_faults_are_caught(self):
        self.assertEqual(self.run_stub("--substitute-empty")["K06"], "fail")
        self.assertEqual(self.run_stub("--norm-mismatch")["K08"], "fail")
        self.assertEqual(self.run_stub("--short-chunk")["K03"], "fail")
        self.assertEqual(self.run_stub("--unseeded")["K04"], "fail")
        st = self.run_stub("--deterministic")
        self.assertEqual(st["K05"], "pass")  # declared non-stochastic and measured so; gate floor must be settle_obs


if __name__ == "__main__":
    unittest.main()
