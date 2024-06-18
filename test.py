from switcher import switch, XP
from numpy.typing import ArrayLike 

def my_fun(x) -> ArrayLike:
    array = switch.xp.asarray(x)
    print(type(array))
    return array

x = my_fun([10])

with switch.set_context(XP.Jax):
    x = my_fun([10])
    print(x.at[0].set(0))

with switch.set_context(XP.Torch):
    my_fun([10])

my_fun([20])