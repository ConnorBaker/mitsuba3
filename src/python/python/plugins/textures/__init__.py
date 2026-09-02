# Import/re-import all files in this folder to register Python plugins
import mitsuba as mi
import sys

if mi.variant() is not None and not mi.variant().startswith('scalar'):
    import importlib

    # `_base.TextureBase` subclasses the variant-specific `mi.Field`, so it
    # must be rebuilt against the newly active variant before any plugin
    # module (re)derives from it.
    from . import _base
    importlib.reload(_base)

    # List of submodules to import
    submodules = [
        'ambient_occlusion',
        'brightness_contrast',
        'bump',
        'blender_blackbody',
        'blender_geometry',
        'blender_image',
        'clamp',
        'color_ramp',
        'combine_color',
        'gradient',
        'curves',
        'hue_saturation',
        'invert_color',
        'layer_weight',
        'map_range',
        'mapping',
        'math',
        'mesh_attribute_adapter',
        'mix_color',
        'noise',
        'noise_texture',
        'normalmap',
        'rgb_to_bw',
        'separate_rgb',
        'texture_coordinate',
        'udim',
        'uv_wrapper',
        'vector_math',
        'white_noise'
    ]

    # Are we importing the submodules for the first time or reloading them?
    reload = submodules[0] in globals()

    module, name = None, None

    for name in submodules:
        module = importlib.import_module(f'.{name}', package=__name__)
        if reload:
            importlib.reload(module)

        # Make the submodule available at package level
        globals()[name] = module

    # A Python plugin registered under a name Mitsuba's C++ already owns does NOT raise:
    # the built-in keeps winning, and the Blender plugin silently never runs. That happened
    # with `blackbody` -- Mitsuba's own is a spectral Planck emitter taking a scalar
    # temperature -- and presented as "expected float, got object" two layers down a scene
    # dict, which reads like a converter bug and is not one. Nothing else in the stack
    # notices, so the collision is asserted here.
    #
    # The list is the plugin names registered by `src/fields/*.cpp` and `src/spectra/*.cpp`
    # (spectra are fields too, and half the collisions live there). It caught a real one
    # at the master port: upstream's `math` expression texture (src/fields/math.cpp) took
    # the name our Blender Math node plugin used to register, which is why that plugin now
    # registers `blender_math`.
    RESERVED_BY_CPP = {
        'bitmap', 'checkerboard', 'constvolume', 'gridvolume', 'math',
        'mesh_attribute', 'sinusoidal', 'volume',
        'blackbody', 'd65', 'irregular', 'rawconstant', 'regular', 'srgb', 'uniform',
    }
    import re as _re
    import pathlib as _pathlib
    _here = _pathlib.Path(__file__).parent
    _clash = []
    for _name in submodules:
        _src = (_here / f'{_name}.py').read_text(encoding='utf-8')
        for _reg in _re.findall(r"register_field\(\s*'([^']+)'", _src):
            if _reg in RESERVED_BY_CPP:
                _clash.append(f'{_name}.py registers {_reg!r}')
    if _clash:
        raise RuntimeError(
            'Blender texture plugin name(s) already owned by a Mitsuba C++ plugin, which '
            'would win silently: ' + '; '.join(_clash))
    del _re, _pathlib, _here, _clash, _name, _src, _reg, RESERVED_BY_CPP

    del importlib, name, submodules, module, reload

del mi, sys
