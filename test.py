from switcher import switch, XP
from numpy.typing import ArrayLike 

def my_fun(x) -> ArrayLike:
    array = switch.xp.asarray(x)
    print(type(array))
    return array

my_fun(10)

with switch.set_context(XP.Jax):
    my_fun(10)

with switch.set_context(XP.Torch):
    my_fun(10)

my_fun(20)