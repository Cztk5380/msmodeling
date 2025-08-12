# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
A module for managing and synchronizing logical time in a multi-threaded environment
"""

import math
import threading
import heapq
from functools import wraps
import time
from typing import Iterator, TypeVar, Generic, List, Optional, Any, Tuple, Callable, Dict, Iterable, Union

import logging
logger = logging.getLogger(__name__)

# Define a generic type variable for type hinting
T = TypeVar('T')


# 1. Global threading state
class ThreadState:
    def __init__(self, thread: threading.Thread):
        self.thread = thread
        self.is_running = True
        self.ts = 0

# native_id -> ThreadState
_all_threads: Dict[int, ThreadState] = {}


def _wait(thread_id=-1):
    if thread_id == -1:
        thread_id = threading.current_thread().native_id
    _all_threads[thread_id].is_running = False


def _awake(thread_id=-1):
    if thread_id == -1:
        thread_id = threading.current_thread().native_id
    _all_threads[thread_id].is_running = True


# 2. Core Time Functions
def now() -> float:
    """
    Returns the current logical timestamp of the calling thread (in seconds).
    """
    if threading.current_thread().native_id not in _all_threads:
        raise ValueError("Thread %d is not tracked by stime", threading.current_thread().native_id)
    return _all_threads[threading.current_thread().native_id].ts


def set_now(ts: float):
    """
    Sets the logical timestamp for current thread.

    This can be used to initialize the main thread's time or manually
    adjust a thread's time. Note: careless use of this function can
    break the causality of the simulation.

    Args:
        ts (float): The logical timestamp (in seconds) to set.
    """
    current_thread = threading.current_thread()
    if current_thread.native_id not in _all_threads:
        state = ThreadState(current_thread)
        _all_threads[current_thread.native_id] = state
    else:
        state = _all_threads[current_thread.native_id]
    state.ts = ts

# Intialize the main thread's timestamp
set_now(0.0)


# Time elapsing mechanisms
def elapse(ts: float):
    """
    Explicitly advance the logical time of the current thread.

    Args:
        ts (float): The logical duration (in seconds) to set.
    """
    if (ts < 0.0):
        raise ValueError("Cannot set negative time")
    set_now(now() + ts)


class DurationDecorator:
    """
    A decorator to specify a duration to elapse after a function call.

    When the decorated function returns, the calling thread's logical time
    is advanced by the specified duration.

    Usage:
        @stime.DurationDecorator(5.0)
        def some_long_running_function():
            ...
    
    Args:
        ts (float): The logical duration (in seconds) to set.
    """
    def __init__(self, ts: float):
        if (ts < 0.0):
            raise ValueError("Cannot set negative time")
        self._duration = ts

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            elapse(self._duration)
            return result
        return wrapper


class Duration:
    """
    A context manager to add a fixed duration to the execution time of a block of code.

    Upon exiting the 'with' block, the current thread's timestamp is
    advanced by the specified duration.

    Usage:
        with stime.Duration(2.5):
            # This block of code logically takes 2.5 seconds.
            ...

    Args:
        ts (float): The logical duration (in seconds) to set.
    """
    def __init__(self, ts: float):
        if (ts < 0.0):
            raise ValueError("Cannot set negative time")
        self._duration = ts

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapse(self._duration)


# 4. Thread Time Management
class Thread(threading.Thread):
    """
    A wrapper for 'threading.Thread' that automatically manages logical timestamps.

    - A new thread inherits the logical timestamp from its parant thread at the
    moment 'start()' is called.
    - When the thread's function exits, its final logical timestamp is recorded.
    - When 'join()' is called, the calling thread's timestamp is synchronized with
    the joined thread's exit timestamp to maintain causality (update to the maximum of the two).
    
    """
    def __init__(self, group: None = None, target: Optional[Callable[..., Any]] = None,
                 name: Optional[str] = None, args: Iterable[Any] = (), kwargs: Optional[Dict[str, Any]] = None, *,
                 daemon: Optional[bool] = None):
        self._original_target = target
        self._original_args = args
        self._original_kwargs = kwargs or {}
        self._exit_timestamp: Optional[float] = None
        self._start_timestamp: float = 0.0

        # The superclass __init__() method will call self._wrapped_run()
        super().__init__(group=group, target=self._wrapped_run, name=name, daemon=daemon)

    def start(self) -> None:
        """
        Starts the thread's activity.

        Sets the new thread's initial timestamp to the current timestamp of the
        calling thread at the moment of starting.
        """
        # Inherit the timestamp of the calling thread.
        self._start_timestamp = now()
        super().start()

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Wait until the thread terminates.

        After the thread terminates, the callling thread's timestamp is updated
        to the maximum of the exited thread's timestamp and the calling thread's timestamp.
        ensure casuality.
        """
        super().join(timeout)
        # Don't sync time if the join stime out and the thread is still alive.
        if not self.is_alive() and self._exit_timestamp is not None:
            caller_now = now()
            # Maintain casuality caller's time can only move forward.
            if self._exit_timestamp > caller_now:
                set_now(self._exit_timestamp)

    def _wrapped_run(self) -> None:
        """
        Internal wrapper for the user's target function to manage the timestamp lifecycle.
        """
        # 1. set the initial timestamp inherited from the parent thread
        set_now(self._start_timestamp)
        try:
            # 2. run the user's target function
            if self._original_target:
                self._original_target(*self._original_args, **self._original_kwargs)
        finally:
            # 3. record the exit timestamp in a 'finally' block to ensure it's always executed
            self._exit_timestamp = now()
            # TOBEDONE consider encapsulating this
            _all_threads.pop(threading.current_thread().native_id)

    
class QueueEmpty(Exception):
    """Exception raised when trying to get an item from an empty queue."""
    pass


class Condition:
    """
    A condition that tracks the running status of a waiting thread.
    """
    def __init__(self, lock=None):
        self._condition = threading.Condition(lock)
        self.waiters = set()

    def __enter__(self):
        return self._condition.__enter__()

    def __exit__(self, *args):
        return self._condition.__exit__(*args)

    def wait(self):
        self._do_wait()

    def wait_for(self, predicate):
        self._do_wait(predicate)

    def notify_all(self):
        for thread_id in self.waiters:
            _awake(thread_id)
        self._condition.notify_all()

    def acquire(self):
        self._condition.acquire()

    def release(self):
        self._condition.release()

    def _do_wait(self, predicate=None):
        _wait()
        self.waiters.add(threading.current_thread().native_id)
        if predicate:
            def wrapped_predicate():
                is_true = predicate()
                if not is_true:
                    _wait()
                return is_true
            self._condition.wait_for(wrapped_predicate)
        else:
            self._condition.wait()
        self.waiters.remove(threading.current_thread().native_id)
        _awake() # make sure we are indeed awake


 # 5. Thread-safe Time-Synchronizing Priority Queue
class Queue(Generic[T]):
    """"
    A priority queue for passing data and synchronizing logical time ordered by timestamp.

    Internalliy uses Python's 'heapq' module to ensure the item with the earliest timestamp is always
    processed first. A counter is used as tie-breaker to handle non-comparable item with the same timestamp.
    """
    def __init__(self, allow_anti_causality_put=False) -> None:
        """
        Args:
            allow_anti_causality_put: If True, allow putting items into the queue with
            a timestamp that is earlier than the earliest item in the queue. This occurs
            when multiple producers put items concurrently into the queue, and their logical
            time is not synchronized and ordered. The consumer should take care of this unordered
            items without breaking the causality. See 'wait_till_due' for detail.
        """
        self._condition = Condition()
        self._shutdown = threading.Event()
        # the heap stores tuple of (timestamp, count, item)
        self._heap: List[Tuple[float, int, T]] = []
        # A counter to act as a tie-breaker for items with the same timestamp.
        self._counter = 0
        self.allowed_earliest_ts = 0
        self.allow_anti_causality_put = allow_anti_causality_put

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self._heap)

    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self._heap)

    def __iter__(self) -> Iterator[T]:
        """
        Iterator over the items in the queue, without timestamp.

        Note: the items are yielded in order of increasing timestamp.
        """
        snapshot = []
        with self._condition:
            for _, _, item in sorted(self._heap):
                snapshot.append(item)
        return iter(snapshot)

    def put(self, item: T) -> None:
        """
        Puts an item into the queue, ordered by current thread's timestamp.

        Args:
            item: The item to put into the queue.
        """
        with self._condition:
            self._put_item(item)
            self._condition.notify_all()

    def put_items(self, items: Union[list, tuple]) -> None:
        """
        Puts a list or tuple of items into the queue, ordered by current thread's timestamp.
        
        Args:
            items: The items to put into the queue.
        """

        with self._condition:
            for item in items:
                self._put_item(item)
            self._condition.notify_all()

    def get(self, block=True) -> T:
        """
        Removes and returns the item with the earliest timestamp.
        If the queue is empty and the 'block' arg is True, the caller is blocked until
        an item is available. 
        Otherwise, if the queue is empty and the 'block' arg is False,
        'QueueEmpty' exception is raised.

        Args:
            block (bool): If True, the caller is blocked until an item is available.
            NEED: add notes and support timeout

        Returns:
            the data item with the earliest timestamp

        Raises:
            QueueEmpty: if the queue is empty and 'block' is False
        """
        with self._condition:
            while not self._heap:
                if block:
                    self._condition.wait()
                else:
                    raise QueueEmpty()
                    
            item_ts, _count, item = heapq.heappop(self._heap)

            # Time synchonize: if the item is in the future, advance the clock
            current_ts = now()
            if item_ts > current_ts:
                set_now(item_ts)
                if not self.allow_anti_causality_put:
                    self.allowed_earliest_ts = now()

            return item

    def peek_due(self) -> Optional[T]:
        """
        Return the item with the earliest timestamp, without removing it.
        But only if its timestamp is less than or equal to the current thread's logical time.

        Returns:
            The earliest item if it is due, otherwise None.
        """
        with self._condition:
            if not self._heap:
                return None
            
            item_ts, _, item = self._heap[0]
            if item_ts <= now():
                return item
            return None

    def peek_all_due(self) -> List[T]:
        """
        Returns a list of all items that are due without removing them from the heap.
        'due' == item's timestamp is less than or equal to the current thread's time.

        Note: This operation is O(n).

        Returns:
            A list of all items that are due.
        """
        with self._condition:
            current_ts = now()
            return [item for item_ts, _, item in self._heap if item_ts <= current_ts]

    def get_due(self) -> Optional[T]:
        """
        Removes and returns the item with the earliest timestamp that is due.
        'due' == item's timestamp is less than or equal to the current thread's time.

        Returns:
            The item if it is due, otherwise None.
        """
        with self._condition:
            if not self._heap:
                return None
                
            if self._heap[0][0] <= now():
                _, _, item = heapq.heappop(self._heap)
                # No time synchronize needed since item_ts <= now()
                return item
            else:
                return None

    def get_all_due(self) -> List[T]:
        """
        Removes and returns all the items that is due.
        'due' == item's timestamp is less than or equal to the current thread's time.

        Returns:
            The item if it is due, otherwise None.
        """
        with self._condition:
            current_ts = now()
            result: List[T] = []
            while self._heap and self._heap[0][0] <= current_ts:
                _, _, item = heapq.heappop(self._heap)
                result.append(item)
            return result

    def wait_till_due(self, timeout_unit: float = 0):
        """
        Simulate timeout wait, The current timestamp is moved fast-forward with
        multiple units of "timeout_unit" till just passing the earliest item in the queue.

        We carefully checks all the threads in the 'logical-time' system to avoid breaking 
        causality. The algorithm works like below:

            Suppoes the earliest timestamp in the queue is t0. We check if all the running
            threads in the system have surpassed t0, meaning there won't be any new items
            to put to the queue by these running ones. We assume the waiting threads would
            always wait and no chance to put items to the queue that would break causality.
            That means we can safely move forward to the earliest item in the queue.

        Args:
            timeout_unit: logical timeout unit in seconds. If timeout_unit is 0, we just move
            forward to the earliest timestamp in the queue.
        """
        with self._condition:
            while not self._heap:
                self._condition.wait()

            current_ts = now()
            first_ts = self._heap[0][0]
            if self.allow_anti_causality_put:
                if first_ts < current_ts:
                    return
            else:
                if first_ts <= current_ts:
                    raise ValueError("Expect now() is earlier but got now: %s >= first: %s" % (current_ts, first_ts))
            self._condition.release()
            try:
                while True:
                    # The following algorithm makes sure we can move forward without breaking
                    # causality by temporarily moving my timestamp forward so that the running
                    # thread with the minimal timestamp can move forward.
                    set_now(first_ts)
                    all_threads = list(_all_threads.values())
                    all_running_ts = [t.ts for t in all_threads if t.is_running]

                    if not all_running_ts:
                        break
                    min_ts = min(all_running_ts)
                    if first_ts > min_ts:
                        time.sleep(0.0001)
                    else:
                        break
            finally:
                self._condition.acquire()
            set_now(current_ts)
            # move fast-forward timeout time to get pass first_ts
            # this simulates the time wait
            if self.allow_anti_causality_put:
                # get the earliest timestamp again just in case some threads
                # put more items
                first_ts = self._heap[0][0]
                if first_ts < current_ts:
                    return
            if timeout_unit == 0:
                waiting_timeout = first_ts - current_ts
            else:
                num_waiting = math.ceil((first_ts - current_ts) / timeout_unit)
                waiting_timeout = num_waiting * timeout_unit
            elapse(waiting_timeout)
            self.allowed_earliest_ts = now()

    def shutdown(self):
        return

    def _put_item(self, item: T) -> None:
        ts = now()
        if ts < self.allowed_earliest_ts:
            raise RuntimeError(f"Items can only be put into the queue at or after {self.allowed_earliest_ts}, "
                               f"but now is {ts}")
        # The counter ensure that even with the same timestamp, the heap
        # has a unique value to compare, preventing errors with non-comparable items.
        entry = (ts, self._counter, item)
        self._counter += 1
        heapq.heappush(self._heap, entry)


def get_logger(logger_name: str):
    class SimulationTimeFilter(logging.Filter):
        def __init__(self):
            super().__init__("sim_time")

        def filter(self, record):
            record.sim_time = now()
            return True    # always return True to ensure the record is processed

    customed_logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler()
    sim_filter = SimulationTimeFilter()
    handler.addFilter(sim_filter)
    formatter = \
        logging.Formatter('[%(sim_time)8.2f][T%(thread)d] %(levelname)-8s %(filename)s:%(lineno)d: %(message)s')
    handler.setFormatter(formatter)
    customed_logger.addHandler(handler)
    customed_logger.propagate = False
    return customed_logger
