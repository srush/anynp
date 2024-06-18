import numpy as np

print("""
class _ArrayAPINameSpace(ModuleType):
""")
for name in dir(np):
    print(
f"""
    {name} = onp.{name}""", end="")
print()