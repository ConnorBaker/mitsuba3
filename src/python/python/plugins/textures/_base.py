# Field-API base class for the Blender texture plugins.
#
# These plugins were written against the pre-Field `mi.Texture` plugin API
# (register_texture + eval/eval_1/eval_3/mean/...). The Field rework unifies
# Texture and Volume into `mi.Field`: plugins now subclass `mi.Field`, declare
# their output/domain contract, and register through `mi.register_field`.
#
# The evaluation surface maps almost 1:1 -- the Field trampoline still
# consults Python overrides named `eval_1`, `eval_3`, `eval_1_grad`, `mean`,
# `max`, `resolution`, `is_spatially_varying`, `traverse` and
# `parameters_changed`. The one genuine rename is spectral `eval`: under the
# Field dispatch, `Field::eval(si, active)` routes through `out_type()` to
# `eval_color3`, so what these plugins used to implement as `eval` is now
# implemented as `eval_color3` (the generic `eval` override slot expects a
# flat FloatStorage of out_dim channels, which is NOT what the old method
# returned).
#
# This module is (re)imported by `textures/__init__.py` BEFORE the plugin
# submodules on every variant change: `mi.Field` is a variant-specific type,
# so the base class must be rebuilt against the newly active variant exactly
# like the plugin classes themselves.

import mitsuba as mi


class TextureBase(mi.Field):
    '''
    Base for texture-style (Color3-valued, surface-domain, JIT-only) Blender
    node plugins. Subclasses implement `eval_color3` / `eval_1` / `eval_3`
    exactly as they implemented `eval` / `eval_1` / `eval_3` under the old
    `mi.Texture` API.
    '''

    def __init__(self, props):
        mi.Field.__init__(self, props)

    def out_type(self):
        return mi.FieldValueType.Color3

    def domain(self):
        return mi.FieldDomain.Surface

    def out_dim(self):
        return 3

    def args_dim(self):
        return 0

    def supports_scalar(self):
        # The plugin registry only imports this package on JIT variants; the
        # implementations use dr.* vectorized arithmetic throughout.
        return False

    def supports_jit(self):
        return True

    def supports_surface_queries(self):
        return True

    def supports_interaction_queries(self):
        return False
