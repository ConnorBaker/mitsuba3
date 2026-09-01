# The hsr port stack

Stacked branches, bottom to top (`git config rebase.updateRefs` is ON in this
repo, so a rebase of an upper branch drags every stacked ref below it):

| branch | contents | moves when |
|---|---|---|
| `upstream-master` | pinned upstream `origin/master` (3989175f at creation) | we deliberately re-pin |
| `pr/1885` | mitsuba3 PR #1885 head, verbatim (`origin/field`, e004b772) | upstream pushes to the PR |
| `pr/1885-fixes` | `pr/1885` + a merge of `upstream-master` with conflicts resolved + adaptation commits (nanobind-3 / Dr.Jit-traversal updates ported onto the PR's replacement files) | either parent moves |
| `hsr/master-port` | our feature work (this branch); split into stacked `hsr/*` topic branches as clusters land | development |

## Refresh procedure

The merge commit in `pr/1885-fixes` must NOT be replayed by a flattening
rebase. When a lower layer moves:

1. Re-pin: `git fetch origin master field`, move `upstream-master` / `pr/1885`.
2. On `pr/1885-fixes`: `git merge upstream-master` (and/or `git merge pr/1885`).
   `rerere` is enabled and has recorded every resolution of the initial merge,
   so unchanged conflicts resolve themselves; only genuinely new conflicts ask
   for work.
3. On `hsr/master-port`: `git rebase pr/1885-fixes`. With `rebase.updateRefs`
   on, any stacked `hsr/*` topic branches that are ancestors of the tip are
   updated in the same pass.

## Why the fixes layer is a merge, not a rebase

The PR's 37 commits patch pre-rewrite code that its own later commits
re-adapt; replaying them onto master means resolving 37 intermediate states
where only the final one survives. The merge resolves the FINAL states once,
`rerere` makes it repeatable, and the result is exactly what a PR author
pushes to make a stale PR mergeable -- so the fixes layer is contributable to
the PR as-is. History stays intact one layer down in `pr/1885`.

## Build status

COMPILES. `nix build '.?submodules=1#mitsuba'` from this branch builds all
three default variants (scalar_rgb, cuda_ad_rgb, cuda_ad_spectral) and the
full fields plugin set, including the ported math plugin. The dependency
triplet (drjit==1.6.0.dev1 at de73a6a2, nanobind==3.0.0 + nanobind-backend
1.0.0 at backend-v1.0.0, split mode ON) is packaged IN-REPO on the
`feat/nix` layer below this branch -- root flake.nix + nix/ -- so the stack
is self-hosting. One compile fix was needed on top of the read-only
adaptation: nanobind 3 added a str_hash argument to detail::ticket, which
the PR's 39 hand-rolled ticket sites in field_v.cpp predate (see the
`field_v.cpp: pass the str_hash argument` commit). Not yet done: running the
Python test suite (src/render/tests/test_texture.py etc.) against the built
package, and regenerating docstr.h with resources/mkdocs.py.
