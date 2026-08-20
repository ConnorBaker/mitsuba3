import mitsuba as mi

from mitsuba.scalar_rgb.test.util import fresolver_append_path


def example_mesh(**kwargs):
    return {
        "type" : "ply",
        "filename" : "resources/data/tests/ply/triangle.ply",
        "face_normals" : True,
        **kwargs
    }


@fresolver_append_path
def test01_merge_single_shape(variants_all_backends_once):
    # One shape on its own, no BSDF
    m = mi.load_dict({
        "type": "merge",
        "child1": example_mesh(),
    })
    assert isinstance(m, mi.Mesh)
    assert m.id() == "child1"

    # One shape in a scene, no BSDF
    m = mi.load_dict({
        "type": "scene",
        "parent": {
            "type": "merge",
            "child1": example_mesh(),
        },
    })
    assert len(m.shapes()) == 1
    assert m.shapes()[0].id() == "parent"

    # One shape in a scene, with a BSDF
    m = mi.load_dict({
        "type": "scene",
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "diffuse" }),
        },
    })
    assert len(m.shapes()) == 1
    assert m.shapes()[0].id() == "parent"

    # Non-mesh --> should be just passed through
    m = mi.load_dict({
        "type": "merge",
        "child1": {
            "type": "sphere",
        },
    })
    assert isinstance(m, mi.Shape)
    assert m.id() == "child1"


@fresolver_append_path
def test02_two_shapes(variants_all_rgb):
    # No BSDF --> doesn't merge
    m = mi.load_dict({
        "type": "merge",
        "child1": example_mesh(),
        "child2": example_mesh(),
    })
    assert len(m) == 2
    assert set([m[0].id(), m[1].id()]) == {"child1", "child2"}

    # No BSDF --> doesn't merge
    m = mi.load_dict({
        "type": "merge",
        "child1": example_mesh(bsdf={ "type": "diffuse" }),
        "child2": example_mesh(),
    })
    assert len(m) == 2
    assert set([m[0].id(), m[1].id()]) == {"child1", "child2"}

    # Same BSDF: merge
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
        }
    })
    assert len(m.shapes()) == 1
    assert m.shapes()[0].id() == "parent"

    # Non-mesh --> doesn't merge
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
            "child2": {
                "type": "sphere",
                "bsdf": { "type": "ref", "id": "bsdf1" }
            },
        }
    })
    assert len(m.shapes()) == 2
    assert set([m.shapes()[0].id(), m.shapes()[1].id()]) == {"parent", "child2"}


@fresolver_append_path
def test03_ray_visibility_is_part_of_the_merge_key(variants_all_rgb):
    """Merging must never widen a shape's ray-visibility mask.

    Regression test. `MergeShape::Key` keyed on the BSDF, media, emitter, sensor and
    the vertex-attribute layout, but not on the ray-visibility mask, and
    `Mesh::merge()` rebuilds the merged mesh from a `Properties` that has no spelling
    for the mask -- so the result came back at `RayVisibility::All` even when every
    operand agreed on a narrower one. A `visible_shadow = False` pane silently began
    casting shadows as soon as it fused with a second pane, which is exactly the case
    a Blender export produces (all window glass shares one material).
    """
    all_flags = 0x3f

    def masks(scene):
        return sorted(s.ray_visibility() for s in scene.shapes())

    # Two panes that AGREE on a narrowed mask: they may fuse, but the mask survives.
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   visible_shadow=False),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   visible_shadow=False),
        }
    })
    assert len(m.shapes()) == 1
    assert masks(m) == [all_flags & ~int(mi.RayVisibility.Shadow)]

    # Two panes that DISAGREE: they must not fuse, and both masks survive intact.
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   visible_shadow=False),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
        }
    })
    assert len(m.shapes()) == 2
    assert masks(m) == sorted([all_flags & ~int(mi.RayVisibility.Shadow), all_flags])

    # Control: with no visibility properties at all, the historical behaviour is
    # unchanged -- one fused shape, fully visible.
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
        }
    })
    assert len(m.shapes()) == 1
    assert masks(m) == [all_flags]
