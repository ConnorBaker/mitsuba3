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

    Regression test. `MergeKey` keyed on the BSDF, media, emitter, sensor and
    the record layout, but not on the ray-visibility mask, and
    `Mesh::merge()` rebuilds the merged mesh from a `Properties` that has no spelling
    for the mask -- so the result came back at `RayMask::All` even when every
    operand agreed on a narrower one. A `visible_shadow = False` pane silently began
    casting shadows as soon as it fused with a second pane, which is exactly the case
    a Blender export produces (all window glass shares one material).
    """
    all_flags = int(mi.RayMask.All)

    def masks(scene):
        return sorted(s.visibility_mask() for s in scene.shapes())

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
    assert masks(m) == [all_flags & ~int(mi.RayMask.Shadow)]

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
    assert masks(m) == sorted([all_flags & ~int(mi.RayMask.Shadow), all_flags])

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


@fresolver_append_path
def test04_silhouette_weight_is_part_of_the_merge_key(variants_all_rgb):
    """`silhouette_sampling_weight` has a `Properties` spelling, and `Mesh::merge`
    never wrote it back -- so every fused mesh silently reverted to the 1.0 default,
    re-weighting silhouette sampling in differentiable renders. Two shapes that agree
    on a non-default weight must still fuse AND keep it; two that disagree must split
    rather than abort, since `merge()` throws on the mismatch.
    """
    def weights(scene):
        return sorted(s.silhouette_sampling_weight() for s in scene.shapes())

    # Agreeing non-default weights: one fused shape, weight preserved.
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   silhouette_sampling_weight=0.25),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   silhouette_sampling_weight=0.25),
        }
    })
    assert len(m.shapes()) == 1
    assert weights(m) == [0.25]

    # Disagreeing weights: two shapes, both intact -- and crucially NOT a load error.
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   silhouette_sampling_weight=0.25),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
        }
    })
    assert len(m.shapes()) == 2
    assert weights(m) == [0.25, 1.0]


@fresolver_append_path
def test05_texture_attributes_survive_a_merge(variants_all_rgb):
    """A `Field` child of a shape becomes a texture attribute at LOAD time
    (`Shape`'s constructor), which is exactly when merging runs. `Mesh::merge`
    rebuilds from a `Properties` those were never written into, so the attribute
    vanished from the fused mesh and any BSDF reading it found nothing. Such shapes
    are now excluded from merging, as mesh attributes already were.
    """
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   my_attr={ "type": "checkerboard" }),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" },
                                   my_attr={ "type": "checkerboard" }),
        }
    })
    assert len(m.shapes()) == 2
    for s in m.shapes():
        assert s.has_attribute("my_attr")

    # Control: without the attribute the two still fuse, so the exclusion is
    # narrow rather than a blanket disabling of the optimisation.
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
