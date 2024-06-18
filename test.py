from switcher import switch, XP

def my_fun(x):
    print(type(switch.xp.asarray(x)))


my_fun(10)

with switch.set_context(XP.Jax):
    my_fun(10)

with switch.set_context(XP.Pytorch):
    my_fun(10)

my_fun(20)