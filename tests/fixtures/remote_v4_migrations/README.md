# Remote-v4 migration fixture

The four migration files as they existed at remote commit
`836e8b5f02e2a2a8bc75993c81678c6534ea885a`, copied byte for byte.

They are here so a test can build a *real* database on that lineage
without needing the git object to be fetchable. `test_the_remote_v4_fixture_is_the_real_thing`
checks the vendored `004` against the checksum pinned in
`phase0/lineages.py`, and — when the commit is present in the repository —
against `git show 836e8b5:...` directly, so this directory cannot drift
away from what it claims to be.

Do not edit these files. They are history.
