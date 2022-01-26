from dataclasses import dataclass


@dataclass
class Pool:
    param1: int

@dataclass
class HEPnOS:
    pools: list


# metalgpy
import numpy as np
import metalgpy as mpy

rng = np.random.RandomState(42)

Pool_ = mpy.meta(Pool)
HEPnOS_ = mpy.meta(HEPnOS)


max_pools = 5
num_pools = mpy.Int(1, max_pools)
pools = mpy.List([Pool_(param1=mpy.Int(0, 10)) for i in range(max_pools)], k=num_pools, invariant=True, name="pools")
hepnos_app = HEPnOS_(pools)

for sample_values, sample_app in mpy.sample(hepnos_app, size=2, rng=rng, deepcopy=True):
    instance = sample_app.evaluate()

    print(f"Frozen Program: {sample_app}")
    print(f"Program evaluation: {instance}")

    print()
