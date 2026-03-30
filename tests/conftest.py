import numpy as np
import pytest

import crflux.models as mods

ALL_MODELS = [
    (mods.PolyGonato, (False,)),
    (mods.PolyGonato, (True,)),
    (mods.HillasGaisser2012, ("H3a",)),
    (mods.HillasGaisser2012, ("H4a",)),
    (mods.H3a_polygonato, ("H3a",)),
    (mods.GaisserStanevTilav, ("3-gen",)),
    (mods.GaisserStanevTilav, ("4-gen",)),
    (mods.CombinedGHandHG, ("H3a",)),
    (mods.CombinedGHandHG, ("H4a",)),
    (mods.ZatsepinSokolskaya, ("default",)),
    (mods.ZatsepinSokolskaya, ("pamela",)),
    (mods.GaisserHonda, ()),
    (mods.Thunman, ()),
    (mods.SimplePowerlaw27, ()),
]


def _model_id(param):
    cls, args = param
    name = cls.__name__
    if args:
        name += f"({args[0]})"
    return name


@pytest.fixture(params=ALL_MODELS, ids=[_model_id(p) for p in ALL_MODELS])
def model(request):
    cls, args = request.param
    return cls(*args)


@pytest.fixture
def energy_grid():
    return np.logspace(1, 11, 200)


@pytest.fixture
def energy_grid_short():
    return np.logspace(2, 6, 50)
