#include <mitsuba/render/shape.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(DiscontinuityFlags) {
    auto disc_flags = nb::enum_<DiscontinuityFlags>(m, "DiscontinuityFlags", nb::is_arithmetic(), D(DiscontinuityFlags))
        .def_value(DiscontinuityFlags, Empty)
        .def_value(DiscontinuityFlags, PerimeterType)
        .def_value(DiscontinuityFlags, InteriorType)
        .def_value(DiscontinuityFlags, DirectionLune)
        .def_value(DiscontinuityFlags, DirectionSphere)
        .def_value(DiscontinuityFlags, HeuristicWalk)
        .def_value(DiscontinuityFlags, AllTypes);

    auto shape_types = nb::enum_<ShapeType>(m, "ShapeType", nb::is_arithmetic(), D(ShapeType))
        .def_value(ShapeType, Mesh)
        .def_value(ShapeType, Rectangle)
        .def_value(ShapeType, BSplineCurve)
        .def_value(ShapeType, Cylinder)
        .def_value(ShapeType, Disk)
        .def_value(ShapeType, LinearCurve)
        .def_value(ShapeType, SDFGrid)
        .def_value(ShapeType, Sphere)
        .def_value(ShapeType, Ellipsoids)
        .def_value(ShapeType, EllipsoidsMesh)
        .def_value(ShapeType, Invalid);

    auto ray_visibility = nb::enum_<RayVisibility>(
            m, "RayVisibility", nb::is_arithmetic(),
            "Per-shape ray visibility, one bit per ray type (Blender's Object > Visibility > "
            "Ray Visibility). A shape invisible to a ray type is skipped by rays of that "
            "type. The default is All.")
        .value("Camera", RayVisibility::Camera, "Rays leaving the sensor")
        .value("Diffuse", RayVisibility::Diffuse,
               "Rays continuing after a diffuse scattering event")
        .value("Glossy", RayVisibility::Glossy,
               "Rays continuing after a glossy or specular reflection")
        .value("Transmission", RayVisibility::Transmission,
               "Rays continuing after a transmission event")
        .value("VolumeScatter", RayVisibility::VolumeScatter,
               "Rays continuing after scattering inside a participating medium")
        .value("Shadow", RayVisibility::Shadow,
               "Shadow rays, i.e. the occlusion test of next-event estimation")
        .value("All", RayVisibility::All, "Visible to every ray type (the default)");
}
