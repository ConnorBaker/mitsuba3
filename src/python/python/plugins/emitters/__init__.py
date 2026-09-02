# Import/re-import all files in this folder to register Python emitter plugins
import mitsuba as mi
import sys

if mi.variant() is not None and not mi.variant().startswith('scalar'):
    # List of submodules to import
    submodules = [
        'blender_lamp',
    ]

    # Are we importing the submodules for the first time or reloading them?
    reload = submodules[0] in globals()

    import importlib
    module, name = None, None

    for name in submodules:
        module = importlib.import_module(f'.{name}', package=__name__)
        if reload:
            importlib.reload(module)

        # Make the submodule available at package level
        globals()[name] = module

    # Same silent-collision hazard as the textures registry (see the note there): a
    # Python plugin registered under a name Mitsuba's C++ already owns does not raise --
    # the built-in keeps winning and the Python plugin silently never runs. The list is
    # the plugin names registered by `src/emitters/*.cpp`.
    RESERVED_BY_CPP = {
        'area', 'constant', 'directional', 'directionalarea', 'envmap',
        'point', 'projector', 'spot', 'sunsky', 'timed_sunsky',
    }
    import re as _re
    import pathlib as _pathlib
    _here = _pathlib.Path(__file__).parent
    _clash = []
    for _name in submodules:
        _src = (_here / f'{_name}.py').read_text(encoding='utf-8')
        for _reg in _re.findall(r"register_emitter\(\s*'([^']+)'", _src):
            if _reg in RESERVED_BY_CPP:
                _clash.append(f'{_name}.py registers {_reg!r}')
    if _clash:
        raise RuntimeError(
            'Blender emitter plugin name(s) already owned by a Mitsuba C++ plugin, which '
            'would win silently: ' + '; '.join(_clash))
    del _re, _pathlib, _here, _clash, _name, _src, _reg, RESERVED_BY_CPP

    del importlib, name, submodules, module, reload

del mi, sys
