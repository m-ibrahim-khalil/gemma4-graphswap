"""Core module."""
import functools

from .util import util_fn


class Thing:
    def run(self, x):
        return helper(x) + self.other()

    def other(self):
        return util_fn()


@functools.lru_cache(maxsize=None)
def helper(x):
    def inner():
        return x

    return inner()


async def fetch_async():
    return helper(1)


class Sub(Thing):
    def go(self):
        return self.run(2)


def make_default():
    return 0


class Holder:
    @property
    def value(self):
        return make_default()

    @value.setter
    def value(self, v):
        self._v = v
