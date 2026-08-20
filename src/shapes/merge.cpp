#include <mitsuba/render/mesh.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MergeShape final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape)
    MI_IMPORT_TYPES(BSDF, Medium, Emitter, Sensor, Mesh)

    MergeShape(const Properties &props) {
        // Note: we are *not* calling the `Shape` constructor as we do not
        // want to accept various properties such as `to_world`.
        std::unordered_map<Key, ref<Mesh>, key_hasher> tbl;
        size_t visited = 0, ignored = 0;
        Timer timer;

        for (auto &prop : props.objects()) {
            ref<Object> shape = prop.get<ref<Object>>();
            Mesh *mesh = prop.try_get<Mesh>();

            if (!mesh || mesh->has_mesh_attributes()) {
                m_objects.push_back(shape);
                ignored++;
                continue;
            }

            Key key;
            key.bsdf = mesh->bsdf();
            key.interior_medium = mesh->interior_medium();
            key.exterior_medium = mesh->exterior_medium();
            key.emitter = mesh->emitter();
            key.sensor = mesh->sensor();
            key.has_normals = mesh->has_vertex_normals();
            key.has_texcoords = mesh->has_vertex_texcoords();
            key.has_face_normals = mesh->has_face_normals();
            /* Ray visibility is part of a shape's IDENTITY, not a shading detail: two
               meshes that differ in it are different objects to the transport code. It
               was absent from this key, so a `visible_shadow = false` window pane fused
               with an ordinary wall and the merged mesh -- built from `Properties` that
               never mention the mask -- came back out at RayVisibility::All. The pane
               then blocked the shadow rays it exists to let through, silently, with no
               warning and no error. MEASURED on the Blender splash scene: the interior
               fell 177x (mean radiance 0.004062 -> 0.000023 against an invariant Cycles
               reference of 0.003836) purely by toggling `load_dict(optimize=)`. */
            key.ray_visibility = mesh->ray_visibility();
            /* `Mesh::merge` THROWS on a `flip_normals` mismatch but this key did not
               separate on it, so two same-BSDF meshes differing only in that flag
               hash-collided and aborted the load outright ("the two meshes are
               incompatible") on a scene that is perfectly legal. Splitting them here
               turns a hard failure into the grouping the throw always expected. */
            key.flip_normals = mesh->has_flipped_normals();

            auto [it, success] = tbl.try_emplace(key, mesh);
            if (!success)
                it->second = it->second->merge(mesh);

            visited++;
        }

        for (auto &kv : tbl) {
            if (tbl.size() == 1 && !props.id().empty())
                kv.second->set_id(props.id());

            m_objects.push_back((ref<Object>) kv.second);
        }

        Log(Info, "Collapsed %zu into %zu meshes. (took %s, %zu objects ignored)",
            visited, tbl.size(), util::time_string((float) timer.value()), ignored);
    }

    std::vector<ref<Object>> expand() const override {
        return m_objects;
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    MI_DECLARE_CLASS(MergeShape)

private:
    struct Key {
        const BSDF *bsdf;
        const Medium *interior_medium;
        const Medium *exterior_medium;
        const Emitter *emitter;
        const Sensor *sensor;
        bool has_normals;
        bool has_texcoords;
        bool has_face_normals;
        uint32_t ray_visibility;
        bool flip_normals;

        bool operator==(const Key &o) const {
            return bsdf == o.bsdf &&
                   interior_medium == o.interior_medium &&
                   exterior_medium == o.exterior_medium &&
                   emitter == o.emitter &&
                   sensor == o.sensor &&
                   has_normals == o.has_normals &&
                   has_texcoords == o.has_texcoords &&
                   has_face_normals == o.has_face_normals &&
                   ray_visibility == o.ray_visibility &&
                   flip_normals == o.flip_normals;
        }
    };

    template <typename T, typename... Ts>
    static inline void hash_combine(std::size_t &seed, const T &v, const Ts &...rest) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        (hash_combine(seed, rest), ...);
    }

    struct key_hasher {
        size_t operator()(const Key &k) const {
            size_t seed = 0;
            int flags = (k.has_normals ? 1 : 0) + (k.has_texcoords ? 2 : 0) +
                        (k.has_face_normals ? 4 : 0) + (k.flip_normals ? 8 : 0);
            hash_combine(seed, k.bsdf, k.interior_medium, k.exterior_medium,
                         k.emitter, k.sensor, flags, k.ray_visibility);
            return seed;
        }
    };
private:
    std::vector<ref<Object>> m_objects;
};

MI_EXPORT_PLUGIN(MergeShape)
NAMESPACE_END(mitsuba)
