"""Wall-clock timing helpers.

Timing is kept separate from the detection functions so that detectors only
return detected objects; callers that care about elapsed time opt in here.
"""

import time
from contextlib import contextmanager

@contextmanager
def timed():
    """Context manager yielding a callable that returns elapsed seconds.

    Usage::

        with timed() as elapsed:
            do_work()
        print(elapsed())
    """
    start = time.time()
    end = None

    def elapsed():
        return (end if end is not None else time.time()) - start

    try:
        yield elapsed
    finally:
        end = time.time()
