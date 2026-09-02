#include <mitsuba/render/interaction.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(RayFlags) {
    auto e = nb::enum_<RayFlags>(m, "RayFlags", nb::is_arithmetic(), D(RayFlags))
        .def_value(RayFlags, Minimal)
        .def_value(RayFlags, Shading)
        .def_value(RayFlags, NormalPartials)
        .def_value(RayFlags, Default)
        .def_value(RayFlags, FollowShape)
        .def_value(RayFlags, DetachShape);

    nb::enum_<RayMask>(m, "RayMask", nb::is_arithmetic(), D(RayMask))
        .def_value(RayMask, Camera)
        // Literal docstrings until the next docstr.h regeneration.
        .value("Diffuse", RayMask::Diffuse,
               "Rays continuing after a diffuse scattering event")
        .value("Glossy", RayMask::Glossy,
               "Rays continuing after a glossy/specular reflection")
        .value("Transmission", RayMask::Transmission,
               "Rays continuing after a transmission event")
        .value("VolumeScatter", RayMask::VolumeScatter,
               "Rays continuing after a scattering event inside a participating medium")
        .value("Shadow", RayMask::Shadow,
               "Shadow rays, i.e. the occlusion test of next-event estimation")
        .def_value(RayMask, All);
}
