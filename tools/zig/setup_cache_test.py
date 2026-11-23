"""A card checkout's Zig cache, installs and standalone tool outputs stay outside it, apart from other checkouts'."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BUILD = """const std = @import("std");

pub fn build(b: *std.Build) void {
    const scratch = b.createModule(.{ .root_source_file = b.path("scratch.zig"), .target = b.graph.host });
    b.installArtifact(b.addExecutable(.{ .name = "scratch", .root_module = scratch }));
    b.step("test", "").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = scratch })).step);
}
"""
# std.testing.tmpDir writes under the working directory's .zig-cache whatever ZIG_LOCAL_CACHE_DIR says.
SCRATCH = """const std = @import("std");

pub fn main() void {}

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

    def test_setup_rejects_cache_paths_inside_checkout(self):
        root = self.checkout("card")
        inside = root / "nested/cache"
        alias = self.work / "alias"
        (root / "nested").mkdir()
        alias.symlink_to(root / "nested", target_is_directory=True)
        for path in (root, inside, alias / "cache"):
            result = subprocess.run([root / "tools/zig/setup_cache.sh"],
                                    env={**self.env, "ZIG_LOCAL_CACHE_DIR": str(path)},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the checkout", result.stderr)
            self.assertFalse((root / ".zig-cache").exists())
            self.assertFalse((root / "zig-out").exists())

    def test_setup_rejects_relative_cache_override(self):
        root = self.checkout("card")
        result = subprocess.run([root / "tools/zig/setup_cache.sh"], cwd=root,
                                env={**self.env, "ZIG_LOCAL_CACHE_DIR": "nested/cache"},
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be absolute", result.stderr)
        self.assertFalse((root / ".zig-cache").exists())

    def test_setup_rejects_symlinked_output(self):
        root = self.checkout("card")
        digest = subprocess.check_output(["shasum", "-a", "256"], input=str(root), text=True).split()[0][:16]
        target = self.cache / "checkouts" / digest
        target.mkdir(parents=True)
        (target / "zig-out").symlink_to(root / "nested", target_is_directory=True)
        result = self.setup_cache(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlinked", result.stderr)
        self.assertFalse((root / ".zig-cache").exists())

    def test_direct_entry_points_default_outside_checkout(self):
        root = self.checkout("card")
        home = self.work / "home"
        home.mkdir()
        stubs = self.work / "stubs"
        stubs.mkdir()
        for name, text in (("zig", STUB_ZIG), ("xcrun", "#!/bin/sh\necho /sdk\n")):
            (stubs / name).write_text(text)
            (stubs / name).chmod(0o755)
        env = {key: value for key, value in self.env.items() if key not in ("ZIG_LOCAL_CACHE_DIR", "ZIG_GLOBAL_CACHE_DIR")}
        env.update(HOME=str(home), ZIG=str(stubs / "zig"), PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}")
        for tool in TOOLS:
            result = subprocess.run(["sh", root / f"tools/zig/{tool}_build.sh"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        cache = home / ".cache/tensorfold/zig"
        self.assertEqual(len(list(cache.glob("checkouts/*/tf-*"))), len(TOOLS))
        self.assertFalse((root / ".zig-cache").exists())
        self.assertFalse((root / "zig-out").exists())

        release = root / "tools/release"
        release.mkdir()
        shutil.copy(ROOT / "tools/release/check_package.py", release / "check_package.py")
        result = subprocess.run(["python3", release / "check_package.py", "--path-checker", str(stubs / "zig")],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)  # The fixture checkout has no package inputs.
        self.assertEqual(len(list(cache.glob("package-repeat-*"))), 1)
        self.assertFalse((root / ".zig-cache").exists())

    def test_plain_build_from_a_clean_checkout_writes_only_outside_it(self):
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
        for step in ("install", "test"):
            result = subprocess.run([zig, "build", step], cwd=root, env=plain, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({path for path in root.rglob("*")} - before, {root / ".zig-cache", root / "zig-out"})
        self.assertEqual(os.readlink(root / ".zig-cache"), str(self.cache))
        self.assertTrue(any((self.cache / "o").iterdir()), "the build's objects are in the shared cache")
        self.assertTrue((self.cache / "tmp").is_dir(), "the test's scratch directory is in the shared cache")
        installed = (root / "zig-out").resolve()
        self.assertEqual(installed.parent.parent, self.cache / "checkouts")
        self.assertTrue((installed / "bin/scratch").is_file(), "the default install is in the shared cache")

    def test_setup_keeps_an_existing_folder_or_link(self):
        for name in (".zig-cache", "zig-out"):
            kept = self.checkout(f"built-{name}")
            (kept / name).mkdir()
            (kept / name / "object").write_text("built")
            linked = self.checkout(f"linked-{name}")
            (linked / name).symlink_to(self.work / "elsewhere")
            for root in (kept, linked):
                self.assertNotEqual(self.setup_cache(root).returncode, 0)
            self.assertEqual((kept / name / "object").read_text(), "built")
            self.assertEqual(os.readlink(linked / name), str(self.work / "elsewhere"))

    def test_concurrent_builds_from_two_checkouts_keep_their_own_outputs(self):
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
        for root in roots:
            self.assertEqual(self.setup_cache(root).returncode, 0)
            self.assertEqual(((root / "zig-out").resolve().parent / "tf-bf16-topk").read_text().strip(), str(root))


if __name__ == "__main__":
    unittest.main()
