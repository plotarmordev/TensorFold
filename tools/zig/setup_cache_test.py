"""A card checkout's Zig cache and standalone tool outputs stay outside it, apart from other checkouts' outputs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BUILD = """const std = @import("std");

pub fn build(b: *std.Build) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("scratch.zig"),
        .target = b.graph.host,
    }) });
    b.step("test", "").dependOn(&b.addRunArtifact(tests).step);
}
"""
# std.testing.tmpDir writes under the working directory's .zig-cache whatever ZIG_LOCAL_CACHE_DIR says.
SCRATCH = """const std = @import("std");

test "scratch file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "scratch", .data = "" });
}
"""
STUB_ZIG = """#!/bin/sh
[ "$1" = version ] && { echo 0.17.0; exit; }
for argument; do
    case $argument in -femit-bin=*) pwd > "${argument#-femit-bin=}" ;; esac
done
"""
TOOLS = ("bf16_topk", "draft_ops", "tree_round")


class SetupCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tensorfold-cache-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        self.cache = self.work / "shared"
        self.env = {**os.environ, "ZIG_LOCAL_CACHE_DIR": str(self.cache)}

    def checkout(self, name):
        root = self.work / name
        (root / "tools/zig").mkdir(parents=True)
        for script in ("setup_cache.sh", *(f"{tool}_build.sh" for tool in TOOLS)):
            shutil.copy(ROOT / "tools/zig" / script, root / "tools/zig" / script)
        return root

    def setup_cache(self, root):
        return subprocess.run([root / "tools/zig/setup_cache.sh"], env=self.env, capture_output=True, text=True)

    def test_plain_build_from_a_clean_checkout_writes_only_to_the_shared_cache(self):
        zig = os.environ.get("ZIG") or shutil.which("zig")
        if not zig:
            self.skipTest("no Zig compiler found (set ZIG, or put zig on PATH)")
        root = self.checkout("card")
        (root / "build.zig").write_text(BUILD)
        (root / "scratch.zig").write_text(SCRATCH)
        before = {path for path in root.rglob("*")}
        for _ in range(2):
            result = self.setup_cache(root)
            self.assertEqual(result.returncode, 0, result.stderr)
        plain = {key: value for key, value in self.env.items() if key != "ZIG_LOCAL_CACHE_DIR"}
        plain["ZIG_GLOBAL_CACHE_DIR"] = str(self.work / "global")
        result = subprocess.run([zig, "build", "test"], cwd=root, env=plain, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({path for path in root.rglob("*")} - before, {root / ".zig-cache"})
        self.assertEqual(os.readlink(root / ".zig-cache"), str(self.cache))
        self.assertTrue(any((self.cache / "o").iterdir()), "the build's objects are in the shared cache")
        self.assertTrue((self.cache / "tmp").is_dir(), "the test's scratch directory is in the shared cache")

    def test_setup_keeps_an_existing_cache_or_link(self):
        kept = self.checkout("built")
        (kept / ".zig-cache").mkdir()
        (kept / ".zig-cache/object").write_text("built")
        linked = self.checkout("linked")
        (linked / ".zig-cache").symlink_to(self.work / "elsewhere")
        for root in (kept, linked):
            self.assertNotEqual(self.setup_cache(root).returncode, 0)
        self.assertEqual((kept / ".zig-cache/object").read_text(), "built")
        self.assertEqual(os.readlink(linked / ".zig-cache"), str(self.work / "elsewhere"))

    def test_concurrent_tool_builds_from_two_checkouts_keep_their_own_executables(self):
        stubs = self.work / "stubs"
        stubs.mkdir()
        for name, text in (("zig", STUB_ZIG), ("xcrun", "#!/bin/sh\necho /sdk\n")):
            (stubs / name).write_text(text)
            (stubs / name).chmod(0o755)
        env = {**self.env, "ZIG": str(stubs / "zig"), "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}"}
        roots = [self.checkout(name) for name in ("first", "second")]
        builds = [subprocess.Popen(["sh", root / f"tools/zig/{tool}_build.sh"], env=env) for tool in TOOLS for root in roots]
        for build in builds:
            self.assertEqual(build.wait(timeout=60), 0)
        for tool in TOOLS:
            outputs = self.cache.glob(f"checkouts/*/tf-{tool.replace('_', '-')}")
            self.assertEqual(sorted(Path(output.read_text().strip()) for output in outputs), roots)


if __name__ == "__main__":
    unittest.main()
