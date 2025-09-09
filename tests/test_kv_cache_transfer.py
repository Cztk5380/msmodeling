# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import threading
import time
import unittest
from kv_cache_transfer import KVTransfer
from stime import Thread


# A tiny mock for the Request object
class Request:
    def __init__(self, req_id):
        self.id = req_id


# ------------------------------------------------------------------
# Single-threaded basic unit-tests
# ------------------------------------------------------------------
class TestKVTransferSingleThreaded(unittest.TestCase):
    """Basic functional tests executed in a single thread."""

    def setUp(self):
        """Create a fresh KVTransfer instance for every test."""
        self.kv = KVTransfer()

    def test_send_and_check(self):
        """After send(), check() must return True for that request."""
        req = Request(123)
        self.assertFalse(self.kv.check(req.id))
        self.kv.send(req.id, foo="bar")
        self.assertTrue(self.kv.check(req.id))

    def test_get_msg_found(self):
        """get_msg() must retrieve the value previously stored."""
        req = Request(456)
        self.kv.send(req.id, key1="value1", key2="value2")
        self.assertEqual(self.kv.get_msg(req.id, "key1"), "value1")
        self.assertEqual(self.kv.get_msg(req.id, "key2"), "value2")

    def test_get_msg_not_found(self):
        """get_msg() must return None for non-existing request or key."""
        req = Request(789)
        self.assertIsNone(self.kv.get_msg(req.id, "whatever"))
        self.kv.send(req.id, x=1)
        self.assertIsNone(self.kv.get_msg(req.id, "nonexistent"))

    def test_remove(self):
        """remove() must delete the entry and subsequent operations behave accordingly."""
        req = Request(555)
        self.kv.send(req.id, data="to-be-deleted")
        self.assertTrue(self.kv.check(req.id))
        self.kv.remove(req.id)
        self.assertFalse(self.kv.check(req.id))
        self.assertIsNone(self.kv.get_msg(req.id, "data"))

    def test_send_duplicate_raises(self):
        """Attempting to send the same request id twice must raise ValueError."""
        req = Request(999)
        self.kv.send(req.id, x=1)
        with self.assertRaises(ValueError):
            self.kv.send(req.id, x=2)


# ------------------------------------------------------------------
# Multi-threaded unit-tests
# ------------------------------------------------------------------
class TestKVTransferMultiThreaded(unittest.TestCase):
    def setUp(self):
        """Create a fresh KVTransfer instance for every test."""
        self.kv = KVTransfer()

    # --------------------------------------------------------------
    # 1. Concurrent send() must never insert duplicates
    # --------------------------------------------------------------
    def test_concurrent_send_no_duplicates(self):
        """Spawn many threads that race to insert the same request id.
        Exactly one thread must succeed, all others must receive
        ValueError.  The final state must contain exactly one entry.
        """
        req = Request(42)
        errors = []
        success_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal success_count
            try:
                self.kv.send(req.id, key="value")
                with lock:
                    success_count += 1
            except ValueError:
                with lock:
                    errors.append("duplicate")

        threads = [Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread succeeded
        self.assertEqual(success_count, 1)
        # All other threads raised ValueError
        self.assertEqual(len(errors), 29)
        # State is consistent: exactly one stored message
        self.assertTrue(self.kv.check(req.id))

    # --------------------------------------------------------------
    # 2. Concurrent readers while writers mutate
    # --------------------------------------------------------------
    def test_concurrent_readers_and_writers(self):
        """Multiple readers repeatedly query an entry while multiple
        writers concurrently add and remove other entries.  Readers
        must never observe partial or corrupted data.
        """
        req_even = [Request(i) for i in range(0, 200, 2)]
        req_odd = [Request(i) for i in range(1, 200, 2)]

        stop = threading.Event()

        # Writer threads: keep adding/removing odd ids
        def writer():
            idx = 0
            while not stop.is_set():
                r = req_odd[idx % len(req_odd)]
                try:
                    self.kv.send(r.id, data="odd")
                except ValueError:
                    pass  # duplicate, ignore
                self.kv.remove(r.id)
                idx += 1

        # Reader threads: only read even ids (should always exist)
        results = []

        def reader():
            while not stop.is_set():
                for r in req_even:
                    if self.kv.check(r.id):
                        val = self.kv.get_msg(r.id, "data")
                        with threading.Lock():
                            results.append(val)
                time.sleep(0)  # yield, increase interleaving

        # Pre-populate even ids
        for r in req_even:
            self.kv.send(r.id, data="even")

        # Start 5 writers and 5 readers
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=writer))
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()

        # Let them race for a short period
        time.sleep(0.1)
        stop.set()
        for t in threads:
            t.join()

        # Readers should have observed only valid values
        for v in results:
            self.assertIn(v, {"even", None})

    # --------------------------------------------------------------
    # 3. Stress test: massive concurrent send/get/remove
    # --------------------------------------------------------------
    def test_massive_concurrent_operations(self):
        """Hammer the store with thousands of concurrent operations:
        send, get and remove on unique keys.  At the end, the store
        must be empty and all operations must have behaved correctly.
        """
        num = 1000
        requests = [Request(i) for i in range(num)]
        barrier = threading.Barrier(num)  # synchronize start
        outcomes = [None] * num

        def worker(idx):
            r = requests[idx]
            barrier.wait()  # all threads start together
            try:
                self.kv.send(r.id, val=idx)
                v = self.kv.get_msg(r.id, "val")
                self.kv.remove(r.id)
                outcomes[idx] = v
            except Exception as e:
                outcomes[idx] = e

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All outcomes must be the index that was inserted
        for i, res in enumerate(outcomes):
            self.assertEqual(res, i, f"Mismatch at index {i}")

        # Store must be empty
        self.assertEqual(len(self.kv.req_id2msg), 0)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()