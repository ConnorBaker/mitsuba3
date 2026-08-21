#include <mitsuba/core/util.h>
#include <mitsuba/render/mesh.h>
#include <drjit/color.h>
#include <array>

// Blender Mesh format types for the exporter
NAMESPACE_BEGIN(blender)
    /// Smooth shading flag
    static const int ME_SMOOTH = 1 << 0;

    /// Triangle tessellation of the mesh, contains references to 3 MLoop and the "real" face
    struct MLoopTri {
        unsigned int tri[3];
        unsigned int poly;
    };

    struct MLoopUV {
        float uv[2];
        int flag;
    };

    struct MLoopCol {
        unsigned char r, g, b, a;
    };

    struct MLoop {
        /// Vertex index.
        unsigned int v;
        /// Edge index.
        unsigned int e;
    };

    /// Contains info about the face, like material ID and smooth shading flag
    struct MPoly {
        /// Offset into loop array and number of loops in the face
        int loopstart;
        int totloop;
        short mat_nr;
        char flag, _pad;
    };

    // Vertex data structure for Blender 2.xx
    struct MVertBlender2 {
        // Position
        float co[3];
        // Normal
        short no[3];
        char flag, bweight;
    };

    // Vertex data structure for Blender 3.xx
    struct MVertBlender3 {
        // Position
        float co[3];
        char flag, bweight;
        char padding[2];
    };

NAMESPACE_END(blender)

NAMESPACE_BEGIN(mitsuba)

/**

Blender mesh loader
----------------------------------------------------------

This plugins converts a Blender Mesh to mitsuba's mesh layout. It is used in the Blender exporter Add-on.
It expects as input pointers to Blender mesh data structures, see GeometryExporter.save_mesh
in the mitsuba2-blender addon for an example.
 */

template <typename Float, typename Spectrum>
class BlenderMesh final : public Mesh<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Mesh, m_name, m_bbox, m_to_world, m_vertex_count,
                    m_face_count, m_vertex_positions, m_vertex_normals,
                    m_vertex_texcoords, m_faces, m_face_normals,
                    add_attribute, initialize)
    MI_IMPORT_TYPES()

    using typename Base::MeshAttributeType;
    using typename Base::ScalarSize;
    using typename Base::ScalarIndex;
    using typename Base::InputFloat;
    using typename Base::InputPoint3f;
    using typename Base::InputVector2f;
    using typename Base::InputVector3f;
    using typename Base::InputNormal3f;
    using typename Base::FloatStorage;
    using Version = util::Version;

    using ScalarIndex3 = std::array<ScalarIndex, 3>;

    /**
    * This constructor created a Mesh object from the part of a blender mesh assigned to a certain material.
    * This allows exporting meshes with multiple materials.
    * This method is inspired by LuxCoreRender (https://github.com/LuxCoreRender/LuxCore/blob/master/src/luxcore/pyluxcoreforblender.cpp)
    * \param props Contains counters and pointers to blender's data structures
    */
    BlenderMesh(const Properties &props) : Base(props) {
        auto fail = [&](const char *descr, auto... args) {
            Throw(("Error while loading Blender mesh \"%s\": " + std::string(descr))
                    .c_str(),
                m_name, args...);
        };

        std::vector<std::string> necessary_fields = {"name", "version", "mat_nr", "vert_count", "loop_tri_count", "loops", "loop_tris", "polys", "verts"};
        for (auto &s : necessary_fields) {
            if (!props.has_property(s))
                fail("missing property \"%s\"!", s);
        }

        // Get Blender version, this is used to determine the right data layout
        Version version(props.get<std::string>("version").c_str());

        m_name = props.get<std::string>("name");
        short mat_nr = (short) props.get<int>("mat_nr");
        size_t vertex_count = props.get<int>("vert_count");
        size_t loop_tri_count = props.get<int>("loop_tri_count");

        // Before Blender 3.6, this points to an array of MLoop, after it is just an array of ints
        const int *loops = reinterpret_cast<const int *>(props.get<int64_t>("loops"));
        const blender::MLoop *loops_old = (const blender::MLoop *) loops;

        // Before Blender 3.6, this points to an array of MLoopTri, after it is just an array of ints
        const unsigned int (*tri_loops)[3] = reinterpret_cast<const unsigned int (*)[3]>(props.get<int64_t>("loop_tris"));
        const blender::MLoopTri *tri_loops_old = (const blender::MLoopTri *) tri_loops;

        // Before Blender 3.6, this points to an array of MPoly, after it is just an array of ints
        const int *polys = reinterpret_cast<const int *>(props.get<int64_t>("polys"));
        const blender::MPoly *polys_old = (const blender::MPoly *) polys;

        // Blender 3.4+ layout
        const int *mat_indices = reinterpret_cast<const int *>(props.get<int64_t>("mat_indices", 0));

        // Blender 3.6+ layout
        const bool *sharp_faces = reinterpret_cast<const bool *>(props.get<int64_t>("sharp_face", 0));

        // The type of vertex buffer will depend on the version of blender used.
        const float (*verts)[3] = reinterpret_cast<const float (*)[3]>(props.get<int64_t>("verts"));
        const blender::MVertBlender2 *verts_old_2 = (const blender::MVertBlender2 *) verts;
        const blender::MVertBlender3 *verts_old_3 = (const blender::MVertBlender3 *) verts;

		// Normals are stored in a separate buffer in Blender 3.1+
        const float (*normals)[3] = reinterpret_cast<const float (*)[3]>(props.get<int64_t>("normals", 0));

        // HSR: CORNER (split) normals -- one per LOOP, not one per vertex.
        //
        // `normals` above is Blender's per-VERTEX normal array, and it is the only normal
        // this plugin ever read. Blender and Cycles do not shade from it. They shade from
        // CORNER normals, which differ from the vertex average wherever an edge is marked
        // sharp, a custom split-normal layer exists, or a Smooth-by-Angle modifier has split
        // a crease -- and none of that is expressible one-normal-per-vertex.
        //
        // The consequence was a silently WRONG shading normal, not a missing feature. On the
        // Blender 4.1 splash scene 40 of 148 meshes and 149042 of 15105612 corners (0.99%)
        // are shaded from a normal that is not the one Blender uses, `sharp_face` explaining
        // none of them. It is most visible on `Lamp 13` (884 sharp edges, 3380 split corners,
        // corner-vs-vertex angle up to 50.9 deg): smoothing the cone/bowl seam on a metal
        // sweeps the normal through the mirror direction and paints a bright specular ring
        // that Cycles does not draw.
        //
        // Nothing else here needs to change to carry them: the vertex dedup `Key` below
        // ALREADY discriminates on the normal for smooth faces, and `vertex_map` chains, so
        // feeding it a per-loop normal makes it split the vertices by itself (Lamp 13: 4924
        // verts -> 5816). Flat faces are deliberately left alone -- Cycles shades those from
        // the triangle's geometric normal, which is what the flat branch already does.
        const float (*loop_normals)[3] = reinterpret_cast<const float (*)[3]>(props.get<int64_t>("loop_normals", 0));

        bool has_cols = false;
        std::vector<std::pair<std::string, const blender::MLoopCol *>> cols;
        for (auto &key : props){
            std::string s(key.name());
            if (s.rfind("vertex_", 0) == 0){
                cols.push_back({s, reinterpret_cast<const blender::MLoopCol *>(props.get<int64_t>(s))});
                has_cols = true;
            }
        }

        // HSR: per-VERTEX float3 attributes, keyed by Blender's VERTEX index. The
        // `vertex_*` family above is per-LOOP 8-bit colour, which is neither the right
        // index space nor enough precision for a coordinate.
        //
        // Blender's Generated texture coordinate is the reason this exists. Cycles'
        // `attr_create_generated` (intern/cycles/blender/mesh.cpp) reads the CD_ORCO layer
        // and derives Generated from the UNDEFORMED vertex positions -- so it is NOT an
        // affine function of the positions this plugin receives, which are the modified
        // ones, and no amount of arithmetic on the exported mesh can recover it. It has to
        // travel with the mesh. Because it is a per-vertex attribute it also survives
        // instancing and particle systems, where a per-object transform would be ambiguous.
        std::vector<std::pair<std::string, const float (*)[3]>> vattrs;
        for (auto &key : props) {
            std::string s(key.name());
            if (s.rfind("vert_attr3_", 0) == 0)
                vattrs.push_back({ "vertex_" + s.substr(11),
                                   reinterpret_cast<const float (*)[3]>(
                                       props.get<int64_t>(s)) });
        }
        bool has_vattrs = !vattrs.empty();

        bool has_uvs = props.has_property("uvs");
        const void *uv_ptr = nullptr;
        if (has_uvs)
            uv_ptr = reinterpret_cast<const void *>(props.get<int64_t>("uvs"));
        else
            Log(Warn, "Mesh %s has no texture coordinates!", m_name);

        // Determine whether the object is globally smooth or flat shaded and set the flag accordingly
        // Blender meshes can be partially smooth AND flat (e.g. with edge split modifier)
        // In this case, flat face vertices will be duplicated.
        m_face_normals = true;
        if (version >= Version(3, 6, 0) && sharp_faces == nullptr)
            // In this case, the mesh is globally smooth shaded, no need to go through all vertices
            m_face_normals = false;
        else {
            if (version >= Version(3, 6, 0)) {
                for (size_t tri_loop_id = 0; tri_loop_id < loop_tri_count; tri_loop_id++) {
                    if (!sharp_faces[polys[tri_loop_id]]) {
                        // At least one smooth face, we can't use global face normals
                        m_face_normals = false;
                        break;
                    }
                }
            }
            else {
                for (size_t tri_loop_id = 0; tri_loop_id < loop_tri_count; tri_loop_id++) {
                    if (blender::ME_SMOOTH & polys_old[tri_loops_old[tri_loop_id].poly].flag) {
                        m_face_normals = false;
                        break;
                    }
                }
            }
        }

        // Temporary buffers for vertices, normals, etc.
        std::vector<std::array<InputFloat, 3>> tmp_vertices; // Store as vector for alignment issues
        std::vector<std::array<InputFloat, 3>> tmp_normals; // Same here
        std::vector<InputVector2f> tmp_uvs;
        std::vector<std::vector<InputFloat>> tmp_cols; // And same here
        std::vector<std::vector<InputFloat>> tmp_vattrs; // And same here
        std::vector<ScalarIndex3> tmp_triangles;

        tmp_vertices.reserve(vertex_count);
        if (!m_face_normals)
            tmp_normals.reserve(vertex_count);
        tmp_triangles.reserve(loop_tri_count);

        if (has_uvs)
            tmp_uvs.reserve(vertex_count);
        if (has_cols) {
            tmp_cols.reserve(cols.size());
            for (size_t p = 0; p < cols.size(); p++) {
                tmp_cols.push_back(std::vector<InputFloat>());
                tmp_cols[p].reserve(3 * vertex_count);
            }
        }
        if (has_vattrs) {
            tmp_vattrs.reserve(vattrs.size());
            for (size_t p = 0; p < vattrs.size(); p++) {
                tmp_vattrs.push_back(std::vector<InputFloat>());
                tmp_vattrs[p].reserve(3 * vertex_count);
            }
        }

        ScalarIndex vertex_ctr = 0;

        // Hash map key to define a unique vertex
        struct Key {
            InputNormal3f normal;
            bool smooth{ false };
            size_t poly{ 0 }; // store the polygon face for flat shading, since
                              // comparing normals is ambiguous due to numerical
                              // precision
            InputVector2f uv{ 0.0f, 0.0f };
            bool operator==(const Key &other) const {
                return dr::all(dr::all(smooth ? normal == other.normal : poly == other.poly) &&
                       uv == other.uv);
            }
            bool operator!=(const Key &other) const {
                return dr::all(dr::all(smooth ? normal != other.normal : poly != other.poly) ||
                       uv != other.uv);
            }
        };

        // Hash map entry
        struct VertexBinding {
            Key key;
            // Index of the vertex in the vertex array
            ScalarIndex value{ 0 };
            VertexBinding *next{ nullptr };
            bool is_init{ false };
        };

        // Hash Map to avoid adding duplicate vertices
        std::vector<VertexBinding> vertex_map(vertex_count);

        size_t duplicates_ctr = 0;
        size_t zero_normal_ctr = 0;
        for (size_t tri_loop_id = 0; tri_loop_id < loop_tri_count; tri_loop_id++) {
            int face_id;
            if (version >= Version(3, 6, 0))
                face_id = polys[tri_loop_id];
             else
                face_id = tri_loops_old[tri_loop_id].poly;

            // We only export the part of the mesh corresponding to the given
            // material id
            if (version >= Version(3, 4, 0) && mat_indices != nullptr && mat_indices[face_id] != mat_nr)
                continue;
            else if (version < Version(3, 4, 0)) {
                if (polys_old[face_id].mat_nr != mat_nr)
                    continue;
            }

            ScalarIndex3 triangle;

            int v0, v1, v2;
            if (version >= Version(3, 6, 0)) {
                const unsigned int *tri_loop = tri_loops[tri_loop_id];
                v0 = loops[tri_loop[0]];
                v1 = loops[tri_loop[1]];
                v2 = loops[tri_loop[2]];
            } else {
                const blender::MLoopTri &tri_loop = tri_loops_old[tri_loop_id];
                v0 = loops_old[tri_loop.tri[0]].v;
                v1 = loops_old[tri_loop.tri[1]].v;
                v2 = loops_old[tri_loop.tri[2]].v;
            }

            const float *co_0, *co_1, *co_2;
            if (version <= Version(3, 0, 0)) {
                // Blender 2.xx - 3.0
                co_0 = verts_old_2[v0].co;
                co_1 = verts_old_2[v1].co;
                co_2 = verts_old_2[v2].co;
            } else if (version >= Version(3, 1, 0) && version <= Version(3, 4, 0)) {
                // Blender 3.1 - 3.4
                co_0 = verts_old_3[v0].co;
                co_1 = verts_old_3[v1].co;
                co_2 = verts_old_3[v2].co;
            } else {
                // Blender 3.5+
                co_0 = verts[v0];
                co_1 = verts[v1];
                co_2 = verts[v2];
            }

            dr::Array<InputPoint3f, 3> face_points;
            face_points[0] = InputPoint3f(co_0[0], co_0[1], co_0[2]);
            face_points[1] = InputPoint3f(co_1[0], co_1[1], co_1[2]);
            face_points[2] = InputPoint3f(co_2[0], co_2[1], co_2[2]);

            // The triangle's own geometric normal, computed for every face rather than
            // only for flat ones: it is also the fallback for a vertex whose stored normal
            // is exactly zero (see below).
            const InputVector3f e1 = face_points[1] - face_points[0];
            const InputVector3f e2 = face_points[2] - face_points[0];
            InputNormal3f geo_normal = m_to_world.scalar() * dr::cross(e1, e2);
            const bool degenerate_face = dr::all(geo_normal == 0.f);
            if (!degenerate_face)
                geo_normal = dr::normalize(geo_normal);

            InputNormal3f normal(0.f);
            bool smooth_face;
            if (version >= Version(3, 6, 0)) {
                // Blender 3.6+ layout
                smooth_face = sharp_faces == nullptr || !sharp_faces[face_id];
            } else {
                smooth_face = blender::ME_SMOOTH & polys_old[face_id].flag;
            }
            if (!smooth_face && !m_face_normals) {
                // Flat shading, use per face normals (only if the mesh is not globally flat)
                if (unlikely(degenerate_face))
                    continue; // Degenerate triangle, ignore it
                normal = geo_normal;
            } else if (smooth_face && !m_face_normals && unlikely(degenerate_face)) {
                // HSR: a smooth face whose geometry is degenerate has no usable normal
                // either way, and the fallback below would have nothing to fall back TO.
                // Skipped here, before any of its vertices is emitted, so the vertex buffer
                // never gains an entry no triangle references.
                bool any_zero = false;
                for (int i = 0; i < 3 && !any_zero; i++) {
                    size_t li, vi;
                    if (version >= Version(3, 6, 0)) {
                        li = ((const int *) tri_loops[tri_loop_id])[i];
                        vi = loops[li];
                    } else {
                        li = tri_loops_old[tri_loop_id].tri[i];
                        vi = loops_old[li].v;
                    }
                    if (vi >= vertex_count)
                        fail("reference to invalid vertex %i!", vi);
                    InputNormal3f n;
                    if (version <= Version(3, 0, 0))
                        n = InputNormal3f(verts_old_2[vi].no[0], verts_old_2[vi].no[1],
                                          verts_old_2[vi].no[2]);
                    else if (loop_normals != nullptr)
                        // HSR: read what the shading branch below will read, or this
                        // pre-check tests a normal that is never used.
                        n = InputNormal3f(loop_normals[li][0], loop_normals[li][1],
                                          loop_normals[li][2]);
                    else
                        n = InputNormal3f(normals[vi][0], normals[vi][1], normals[vi][2]);
                    any_zero = dr::all(n == 0.f);
                }
                if (any_zero)
                    continue;
            }

            InputFloat color_factor = dr::rcp(255.f);

            for (int i = 0; i < 3; i++) {
                size_t loop_index, vert_index;
                if (version >= Version(3, 6, 0)) {
                    const int *tri_loop = (const int *) tri_loops[tri_loop_id];
                    loop_index = tri_loop[i];
                    vert_index = loops[loop_index];
                } else {
                    loop_index = tri_loops_old[tri_loop_id].tri[i];
                    vert_index = loops_old[loop_index].v;
                }

                if (unlikely((vert_index >= vertex_count)))
                    fail("reference to invalid vertex %i!", vert_index);

                Key vert_key;
                if (smooth_face && !m_face_normals) {
                    if (version <= Version(3, 0, 0)) {
                        // Blender 2.xx - 3.0
                        const short *no = verts_old_2[vert_index].no;
                        // Store per vertex normals if the face is smooth or if the mesh is globally flat
                        normal = m_to_world.scalar() * InputNormal3f(no[0], no[1], no[2]);
                    } else if (loop_normals != nullptr) {
                        // HSR: the corner normal -- indexed by LOOP, which is what Blender
                        // shades with. See the note at the property read above.
                        const float *no = loop_normals[loop_index];
                        normal = m_to_world.scalar() * InputNormal3f(no[0], no[1], no[2]);
                    } else {
                        const float *no = normals[vert_index];
                        // Store per vertex normals if the face is smooth or if the mesh is globally flat
                        normal = m_to_world.scalar() * InputNormal3f(no[0], no[1], no[2]);
                    }

                    // HSR: an exactly-zero vertex normal used to ABORT the whole mesh.
                    // Blender's own 4.1 splash scene contains them -- a vertex whose
                    // incident faces are all degenerate, or whose face normals cancel --
                    // and Blender renders those meshes without comment, so refusing the
                    // mesh loses far more than the vertex does. The triangle's geometric
                    // normal is substituted, which is the only defined answer available,
                    // and the count is reported rather than swallowed: a mesh that needed
                    // the fallback has bad data and the user should hear about it.
                    if (unlikely(dr::all(normal == 0.f))) {
                        normal = geo_normal;
                        zero_normal_ctr++;
                    } else
                        normal = dr::normalize(normal);
                    vert_key.smooth = true;
                } else {
                    // vert_key.smooth = false (default), flat shading
                    // Store the referenced polygon (face)
                    vert_key.poly = face_id;
                }

                vert_key.normal = normal;

                if (has_uvs) {
                    if (version <= Version(3, 4, 0)) {
                        // Blender 2.xx - 3.4
                        const blender::MLoopUV *uvs = (const blender::MLoopUV *) uv_ptr;
                        const blender::MLoopUV &loop_uv = uvs[loop_index];
                        const InputVector2f uv(loop_uv.uv[0], 1.0f - loop_uv.uv[1]);
                        vert_key.uv = uv;
                    } else {
                        // Blender 3.5+
                        const float (*uvs)[2] = (const float (*)[2]) uv_ptr;
                        const InputVector2f uv(uvs[loop_index][0], 1.0f - uvs[loop_index][1]);
                        vert_key.uv = uv;
                    }
                }

                // Vertex index in the blender mesh is the map index
                VertexBinding *map_entry = &vertex_map[vert_index];
                while (vert_key != map_entry->key && map_entry->next != nullptr)
                    map_entry = map_entry->next;

                if (map_entry->is_init && map_entry->key == vert_key) {
                    triangle[i] = map_entry->value;
                    duplicates_ctr++;
                } else {
                    if (map_entry->is_init) {
                        // Add a new entry
                        map_entry->next = new VertexBinding();
                        map_entry       = map_entry->next;
                    }
                    ScalarSize vert_id = vertex_ctr++;
                    map_entry->key     = vert_key;
                    map_entry->value   = vert_id;
                    map_entry->is_init = true;
                    // Add stuff to the temporary buffers
                    InputPoint3f pt = m_to_world.scalar() * face_points[i];
                    tmp_vertices.push_back({pt.x(), pt.y(), pt.z()});
                    if (!m_face_normals)
                        tmp_normals.push_back({normal.x(), normal.y(), normal.z()});
                    if (has_uvs)
                        tmp_uvs.push_back(vert_key.uv);
                    if (has_cols) {
                        for (size_t p = 0; p < cols.size(); p++) {
                            const blender::MLoopCol &loop_col = cols[p].second[loop_index];
                            // Blender stores vertex colors in sRGB space
                            tmp_cols[p].push_back(dr::srgb_to_linear(loop_col.r * color_factor));
                            tmp_cols[p].push_back(dr::srgb_to_linear(loop_col.g * color_factor));
                            tmp_cols[p].push_back(dr::srgb_to_linear(loop_col.b * color_factor));
                        }
                    }
                    if (has_vattrs) {
                        for (size_t p = 0; p < vattrs.size(); p++) {
                            // Indexed by the BLENDER vertex index, not the loop index, and
                            // NOT transformed by `m_to_world`: these are object-space
                            // quantities by construction.
                            const float *a = vattrs[p].second[vert_index];
                            tmp_vattrs[p].push_back(a[0]);
                            tmp_vattrs[p].push_back(a[1]);
                            tmp_vattrs[p].push_back(a[2]);
                        }
                    }
                    triangle[i] = vert_id;
                }
            }
            tmp_triangles.push_back(triangle);
        }
        Log(Info, "%s: Removed %i duplicates", m_name, duplicates_ctr);
        if (zero_normal_ctr > 0)
            Log(Warn, "%s: %i vertex normal(s) were exactly zero and were replaced by the "
                      "triangle's geometric normal. The source mesh has degenerate geometry.",
                m_name, zero_normal_ctr);

        if (vertex_ctr == 0)
            return;

        m_face_count = (ScalarSize) tmp_triangles.size();
        m_faces = dr::load<DynamicBuffer<UInt32>>(tmp_triangles.data(), m_face_count * 3);

        m_vertex_count = vertex_ctr;
        m_vertex_positions = dr::load<FloatStorage>(tmp_vertices.data(), m_vertex_count * 3);
        if (!m_face_normals)
            m_vertex_normals = dr::load<FloatStorage>(tmp_normals.data(), m_vertex_count * 3);

        if (has_uvs)
            m_vertex_texcoords = dr::load<FloatStorage>(tmp_uvs.data(), m_vertex_count * 2);

        if (has_cols) {
            for (size_t p = 0; p < cols.size(); p++)
                add_attribute(cols[p].first, 3, tmp_cols[p]);
        }

        if (has_vattrs) {
            for (size_t p = 0; p < vattrs.size(); p++)
                add_attribute(vattrs[p].first, 3, tmp_vattrs[p]);
        }

        initialize();
    }

    MI_DECLARE_CLASS(BlenderMesh)
};

MI_EXPORT_PLUGIN(BlenderMesh)
NAMESPACE_END(mitsuba)
