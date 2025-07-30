# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
import time
from queue import Queue as StandardQueue # Used for thread-safe communication in tests

import stime


class NonComparable:
    """A simple class that cannot be compared to other objects, testing purposes only"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'NonComparable({self.name})'


class TestTimedModule(unittest.TestCase):
    def setUp(self):
        # rest the logical time ensure test isolation
        stime.set_now(0)

    # 1. core functionality tests
    
    def test_now_and_set_now(self):
        self.assertAlmostEqual(stime.now(), 0)
        stime.set_now(1)
        self.assertAlmostEqual(stime.now(), 1)

    def test_elapse(self):
        stime.elapse(1)
        self.assertAlmostEqual(stime.now(), 1)

    def test_duration_decorator(self):
        stime.set_now(10)

        @stime.duration(10)
        def decorated_function():
            self.assertAlmostEqual(stime.now(), 10)
            return "done"

        result = decorated_function()
        self.assertEqual(result, "done")
        self.assertAlmostEqual(stime.now(), 20)

    def test_duration_context_manager(self):
        with stime.Duration(1):
            self.assertAlmostEqual(stime.now(), 0)
        self.assertAlmostEqual(stime.now(), 1)

    # 2. Queue tests

    def test_queue_ordering_with_heap(self):
        q = stime.Queue()

        stime.set_now(10.0)
        q.put("item_at_10")
        
        stime.set_now(5.0)
        q.put("item_at_5")
        
        stime.set_now(15.0)
        q.put("item_at_15")
        

        self.assertEqual(len(q), 3)
        self.assertEqual(q.get(), "item_at_5")
        self.assertEqual(q.get(), "item_at_10")
        self.assertEqual(q.get(), "item_at_15")
        self.assertEqual(len(q), 0)

    def test_queue_put_lists(self):
        q = stime.Queue()

        stime.set_now(10.0)
        items = ["item_at_10_1", "item_at_10_2", "item_at_10_3"]
        q.put_items(items)

        stime.set_now(5.0)
        q.put("item_at_5")

        stime.set_now(15.0)
        q.put("item_at_15")

        self.assertEqual(len(q), 5)
        self.assertEqual(q.get(), "item_at_5")
        self.assertEqual(q.get(), "item_at_10_1")
        self.assertEqual(q.get(), "item_at_10_2")
        self.assertEqual(q.get(), "item_at_10_3")
        self.assertEqual(q.get(), "item_at_15")
        self.assertEqual(len(q), 0)

    def test_queue_get_time_synchronization(self):
        q = stime.Queue()

        # Case1: get an item from the future
        stime.set_now(50.0)
        q.put("future_item")

        stime.set_now(10.0)
        item = q.get()
        self.assertEqual(item, "future_item")
        self.assertAlmostEqual(stime.now(), 50.0, "The clock should be advanced to the item's timestamp")

        # Case2: get an item from the past
        stime.set_now(40.0)
        with self.assertRaisesRegex(RuntimeError, "Items can only be put into the queue at or after 50.0"):
            q.put("past_item")

    def test_queue_with_non_comparable_items(self):
        q = stime.Queue()
        item1 = NonComparable("a")
        item2 = NonComparable("b")
        
        stime.set_now(10.0)
        try:
            # This would raise a TypeError if the the tie-breaker was not implemented.
            q.put(item1)
            q.put(item2)
        except TypeError:
            self.fail("TypeError raised when putting non-comparable items with the same timestamp into the queue")

        # Check that they can be retrieved (order may vary due to the counter)
        retrieved1 = q.get()
        retrieved2 = q.get()
        self.assertIn(retrieved1, [item1, item2])
        self.assertIn(retrieved2, [item1, item2])
        self.assertNotEqual(retrieved1, retrieved2)

    def test_queue_due_methods(self):
        q = stime.Queue()
        stime.set_now(30)
        q.put("item_30")
        stime.set_now(40)
        q.put("item_40")
        stime.set_now(50)
        q.put("item_50")
        stime.set_now(60)
        q.put("item_60")
        
        stime.set_now(45)

        self.assertEqual(q.peek_due(), "item_30")
        self.assertEqual(len(q), 4, "peek_due should not remove items from the queue")

        all_due = q.peek_all_due()
        self.assertEqual(all_due, ["item_30", "item_40"])
        self.assertEqual(len(q), 4, "peek_all_due should not remove items from the queue")

        self.assertEqual(q.get_due(), "item_30")
        self.assertEqual(len(q), 3, "get_due should remove items from the queue")
        self.assertEqual(q.get_due(), "item_40")
        self.assertEqual(len(q), 2)
        self.assertIsNone(q.get_due(), "Should return None as item_50 is not due")
        self.assertEqual(len(q), 2)

    def test_get_all_due(self):
        q = stime.Queue()
        stime.set_now(30)
        q.put("item_30")
        stime.set_now(40)
        q.put("item_40")
        stime.set_now(50)
        q.put("item_50")
        stime.set_now(45)

        removed_item = q.get_all_due()
        self.assertCountEqual(removed_item, ["item_30", "item_40"])
        self.assertEqual(len(q), 1)
        self.assertEqual(q.get(), "item_50")

    def test_queue_iterators(self):
        q = stime.Queue()
        stime.set_now(40)
        q.put("item_40")
        stime.set_now(30)
        q.put("item_30")
        stime.set_now(50)
        q.put("item_50")

        items = list(q)
        self.assertEqual(items, ["item_30", "item_40", "item_50"])

    def test_queue_wait_till_due_and_restriction(self):
        q = stime.Queue()
        stime.set_now(10)

        # put an itme in the future at time 27
        # we need to set the time to 27 for the put() call
        stime.set_now(27)
        q.put("item_27")

        # reset the time to 10 before the item, so that we can wait it 
        stime.set_now(10)

        self.assertEqual(q._heap[0][0], 27.0)

        # wait for the item
        q.wait_till_due(timeout_unit=5.0)

        # Expectation:
        # first_ts = 27.0, current_ts = 10.0, timeout_unit = 5.0
        # num_waiting = math.ceil((27 - 10) / 5) = math.ceil(17.0 / 5) = 4
        # elapsed_time = 4 * 5 = 20
        # new_time = current_ts + elapsed_time = 10.0 + 20 = 30.0
        self.assertAlmostEqual(stime.now(), 30.0)
        self.assertAlmostEqual(q.allowed_earliest_ts, 30.0)

        # part2, test the put() restriction after waiting

        stime.set_now(29.9)
        with self.assertRaisesRegex(RuntimeError, 'Items can only be put into the queue at or after 30.0'):
            q.put("illegal_item")

        # action: try to put an item exactly at the allowed_earliest_ts (should succeed)
        stime.set_now(30.0)
        try:
            q.put("legal_item_at_30.0")
        except RuntimeError:
            self.fail("put() raised RuntimeError unexpectedly at allowed_earliest_ts")

        stime.set_now(35)
        try:
            q.put("legal_item_at_35.0")
        except RuntimeError:
            self.fail("put() raised RuntimeError unexpectedly after allowed_earliest_ts")

        # verification now the queue should contain the original and two new legal item
        self.assertEqual(len(q), 3)
        self.assertEqual(q.get(), "item_27")
        self.assertEqual(q.get(), "legal_item_at_30.0")
        self.assertEqual(q.get(), "legal_item_at_35.0")

        # part3 edge case preconditions for the wait_till_due()
        # calling when now() >= first_ts should raise ValueError
        q_fail = stime.Queue()
        stime.set_now(50)
        q_fail.put("item_at_50")
        with self.assertRaises(ValueError):
            q_fail.wait_till_due(5.0)

    def test_queue_wait_till_due_single_item_slow_producer(self):
        """wait_till_due when a single item is put to the queue"""
        q = stime.Queue()

        def slow_producer():
            time.sleep(1)
            stime.set_now(5.0)
            q.put("item_at_5")

        t = stime.Thread(target=slow_producer, daemon=True)
        t.start()
        q.wait_till_due(0.01)
        self.assertEqual(len(q), 1)
        self.assertEqual(q.get(), "item_at_5")
        t.join()

    def test_queue_wait_till_due_two_items_slow_producer(self):
        """wait_till_due when two items are put in the queue"""
        q = stime.Queue()
        stime.set_now(10.0)

        q.put("item_at_10")
        stime.set_now(0)
        
        def slow_producer():
            time.sleep(1)
            stime.set_now(5)
            q.put("item_at_5")
        
        t = stime.Thread(target=slow_producer, daemon=True)
        t.start()
        q.wait_till_due(0.01)
        self.assertEqual(len(q), 2)
        self.assertEqual(q.get(), "item_at_5")
        self.assertEqual(q.get(), "item_at_10")
        t.join()

    def test_queue_wait_till_due_two_waiters(self):
        """wait_till_due whem two threads are waiting"""
        def wait_for_empty():
            q = stime.Queue()
            q.wait_till_due(0.01)

        first_waiter = stime.Thread(target=wait_for_empty, daemon=True)
        first_waiter.start()

        q = stime.Queue()
        stime.set_now(10.0)
        q.put("item_at_10")
        
        stime.set_now(0.0)

        def slow_producer():
            time.sleep(1)
            stime.set_now(5)
            q.put("item_at_5")

        t = stime.Thread(target=slow_producer, daemon=True)
        t.start()
        q.wait_till_due(0.01)
        self.assertEqual(len(q), 2)
        self.assertEqual(q.get(), "item_at_5")
        self.assertEqual(q.get(), "item_at_10")
        t.join()

    # 3. Thread tests
    
    def test_thread_start_time_inheritance(self):
        """Test that a child thread inherits the parent's time at the moment of start()"""
        # use standard queue for thread-safe result passing
        result_queue = StandardQueue()

        def child_thread():
            # record the time as soon as the thread starts
            result_queue.put(stime.now())

        stime.set_now(0)
        parent_thread = stime.Thread(target=child_thread)

        # parent_thread object is created but not started
        stime.set_now(15)
        parent_thread.start()
        parent_thread.join()

        start_time = result_queue.get()
        self.assertAlmostEqual(start_time, 15, "child should inherit time from when start was called")

    def test_thread_join_time_synchronization(self):
        """Test that joining a thread syncs the parent's time correctly"""
        result_queue = StandardQueue()

        def child_thread():
            # child starts, inherits parent time, then advances its own time
            stime.elapse(20)
            result_queue.put("done")

        # case1 parent time should be advanced by join()
        stime.set_now(100)
        child = stime.Thread(target=child_thread)
        child.start()

        # wait for child to finish its work
        result_queue.get()

        child.join()
        self.assertAlmostEqual(stime.now(), 120, "parent's time should sync to child's later exit time")

        # case 2 parent time should NOT GO BACK
        stime.set_now(200)
        child = stime.Thread(target=child_thread)
        child.start()

        stime.elapse(50)
        self.assertAlmostEqual(stime.now(), 250)

        child.join()
        self.assertAlmostEqual(stime.now(), 250, "parent time should not go backward")



if __name__ == "__main__":
    unittest.main()
