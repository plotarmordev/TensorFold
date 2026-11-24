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
FIXTURES = ("attn", "basic", "mamba", "moe", "qmv")
# Imports check_mlx_ops against stub mlx/numpy, compiles its runner with a stub swiftc and stages one batch.
MLX_CHECK = "import check_mlx_ops as c; c.runner(); print(c.Batch('weights').dir)"


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
        scripts = ("setup_cache.sh", "outside_checkout.py", "check_mlx_ops.py", "test_fabric.sh", "cuda_sass_compare.py",
                   "cuda_sass_identity.py", "cuda_sass_sources.py", *(f"{tool}_build.sh" for tool in TOOLS),
                   *(f"xpu_fixtures_{name}.py" for name in FIXTURES))
        for script in scripts:
            shutil.copy(ROOT / "tools/zig" / script, root / "tools/zig" / script)
        (root / "tools/release").mkdir()
        shutil.copy(ROOT / "tools/release/check_package.py", root / "tools/release/check_package.py")
        return root

    def entry_points(self, root, env):
        """Run each direct build script and the package check; return their results."""
        scripts = [["sh", root / f"tools/zig/{tool}_build.sh"] for tool in TOOLS]
        package = ["python3", root / "tools/release/check_package.py", "--path-checker", env["ZIG"]]
        return [subprocess.run(command, env=env, capture_output=True, text=True) for command in (*scripts, package)]

    def setup_cache(self, root, cache=None):
        env = self.env if cache is None else {**self.env, "ZIG_LOCAL_CACHE_DIR": str(cache)}
        return subprocess.run([root / "tools/zig/setup_cache.sh"], env=env, capture_output=True, text=True)

    def stubs(self):
        stubs = self.work / "stubs"
        stubs.mkdir(exist_ok=True)
        for name, text in (("zig", STUB_ZIG), ("xcrun", "#!/bin/sh\necho /sdk\n")):
            (stubs / name).write_text(text)
            (stubs / name).chmod(0o755)
        return {**self.env, "ZIG": str(stubs / "zig"), "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}"}

    def python_stubs(self):
        """Empty numpy and mlx modules, enough to import the generators and the Metal checker without them."""
        stubs = self.work / "pystubs"
        (stubs / "mlx").mkdir(parents=True)
        (stubs / "mlx/__init__.py").write_text("")
        (stubs / "mlx/core.py").write_text("def __getattr__(name):\n    return name\n")
        (stubs / "numpy.py").write_text("")
        return stubs

    def test_setup_rejects_cache_paths_inside_checkout(self):
        root = self.checkout("card")
        inside = root / "nested/cache"
        alias = self.work / "alias"
        (root / "nested").mkdir()
        alias.symlink_to(root / "nested", target_is_directory=True)
        for path in (root, inside, alias / "cache"):
            result = self.setup_cache(root, path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the checkout", result.stderr)
            self.assertFalse((root / ".zig-cache").exists())
            self.assertFalse((root / "zig-out").exists())

    def test_setup_rejects_a_case_variant_of_the_checkout(self):
        root = self.checkout("card")
        variant = Path(str(root).swapcase())
        if not variant.exists():
            self.skipTest("case-sensitive filesystem")
        result = self.setup_cache(root, variant / "nested/cache")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the checkout", result.stderr)
        self.assertEqual(list(root.iterdir()), [root / "tools"])

    def test_entry_points_reject_a_cache_inside_the_checkout(self):
        root = self.checkout("card")
        env = {**self.stubs(), "ZIG_LOCAL_CACHE_DIR": str(root / "nested/cache")}
        for result in self.entry_points(root, env):
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the checkout", result.stderr)
        self.assertEqual(list(root.iterdir()), [root / "tools"])

    def test_every_entry_point_rejects_a_cache_path_with_a_newline(self):
        root = self.checkout("card")
        link = self.work / "link"
        link.symlink_to(f"{root}\n")  # Resolves to a newline only after the link is followed.
        # Command substitution would strip the newline and use the checkout itself.
        for cache in (f"{root}\n", str(link)):
            env = {**self.stubs(), "ZIG_LOCAL_CACHE_DIR": cache}
            for result in (self.setup_cache(root, cache), *self.entry_points(root, env)):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must not contain a newline", result.stderr)
        self.assertEqual(list(root.iterdir()), [root / "tools"])

    def test_metal_checker_stages_runner_and_tensors_outside_checkout(self):
        root = self.checkout("card")
        home = self.work / "home"
        swiftc = self.work / "stubs/swiftc"
        swiftc.parent.mkdir()
        swiftc.write_text('#!/bin/sh\nwhile [ "$1" != -o ]; do shift; done\ntouch "$2"\n')
        swiftc.chmod(0o755)
        env = {key: value for key, value in self.env.items() if key not in ("ZIG_LOCAL_CACHE_DIR", "PYTHONDONTWRITEBYTECODE")}
        env.update(HOME=str(home), PATH=f"{swiftc.parent}{os.pathsep}{os.environ['PATH']}",
                   PYTHONPATH=f"{root / 'tools/zig'}{os.pathsep}{self.python_stubs()}")
        result = subprocess.run(["python3", "-c", MLX_CHECK], cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        staged = Path(result.stdout.strip())
        self.assertEqual(staged.parents[3], home / ".cache/tensorfold/zig/checkouts")
        self.assertTrue(staged.is_dir())
        self.assertTrue((staged.parents[1] / "metal_bench").is_file())
        self.assertEqual(list(root.iterdir()), [root / "tools"])
        # Its own bytecode is written before the flag in check_mlx_ops.py can run; its sibling imports write none.
        self.assertEqual(list(root.rglob("outside_checkout*.pyc")), [])

    def test_sass_compare_defaults_its_output_outside_checkout(self):
        root = self.checkout("card")
        home = self.work / "home"
        env = {key: value for key, value in self.env.items() if key != "ZIG_LOCAL_CACHE_DIR"}
        env["HOME"] = str(home)
        # The fixture checkout lacks the CUDA sources, so the snapshot fails after the output folder is made.
        result = subprocess.run(["python3", "-B", "tools/zig/cuda_sass_compare.py", ".", ".", "--plan-only"],
                                cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        output, = (home / ".cache/tensorfold/zig/sass-compare").iterdir()
        self.assertTrue((output / "summary.json").is_file())
        self.assertIn(str(output), result.stdout)
        self.assertEqual(list(root.iterdir()), [root / "tools"])

    def test_fabric_tests_build_outside_checkout_without_setup(self):
        root = self.checkout("card")
        home = self.work / "home"
        mcdma = self.work / "mcdma"
        (mcdma / "rpc").mkdir(parents=True)
        (mcdma / "rpc/libmcdma_rpc.c").write_text("")
        env = self.stubs()
        (Path(env["ZIG"]).parent / "cc").write_text('#!/bin/sh\nwhile [ "$1" != -o ]; do shift; done\ntouch "$2"\n')
        (Path(env["ZIG"]).parent / "cc").chmod(0o755)
        env = {key: value for key, value in env.items() if key != "ZIG_LOCAL_CACHE_DIR"}
        env.update(HOME=str(home), TF_MCDMA_SRC=str(mcdma))
        result = subprocess.run(["bash", root / "tools/zig/test_fabric.sh"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        fabric, = home.glob(".cache/tensorfold/zig/checkouts/*/fabric")
        self.assertTrue((fabric / "tf-linktest").is_file())
        self.assertEqual(len(list(fabric.glob("libmcdma-rpc.*"))), 1)
        self.assertEqual(list(root.iterdir()), [root / "tools"])

    def test_fixture_generators_default_outside_checkout(self):
        root = self.checkout("card")
        home = self.work / "home"
        env = {key: value for key, value in self.env.items() if key != "TF_FIXTURES_DIR"}
        env.update(HOME=str(home), TF_NEMOTRON_DIR=str(self.work / "no-checkpoint"), PYTHONPATH=str(self.python_stubs()))
        for name in FIXTURES:
            script = root / f"tools/zig/xpu_fixtures_{name}.py"
            result = subprocess.run(["python3", "-B", script], cwd=root, env=env, capture_output=True, text=True)
            self.assertIn("no-checkpoint", result.stderr)  # It stops at the missing checkpoint, after choosing its folder.
        self.assertTrue((home / "tensorfold-fixtures").is_dir())
        self.assertEqual(list(root.iterdir()), [root / "tools"])

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
        env = {key: value for key, value in self.stubs().items() if key not in ("ZIG_LOCAL_CACHE_DIR", "ZIG_GLOBAL_CACHE_DIR", "PYTHONDONTWRITEBYTECODE")}
        env["HOME"] = str(home)
        *builds, package = self.entry_points(root, env)
        for result in builds:
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(package.returncode, 0)  # The fixture checkout has no package inputs.
        cache = home / ".cache/tensorfold/zig"
        self.assertEqual(len(list(cache.glob("checkouts/*/tf-*"))), len(TOOLS))
        self.assertEqual(len(list(cache.glob("package-repeat-*"))), 1)
        self.assertEqual(list(root.iterdir()), [root / "tools"])
        self.assertEqual(list(root.rglob("*.pyc")), [])

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

    def test_setup_refuses_a_link_whose_target_adds_a_trailing_newline(self):
        root = self.checkout("card")
        (root / "nested").mkdir()
        newline = Path(f"{self.cache}\n")  # Command substitution would read this target as the shared cache.
        newline.symlink_to(root / "nested", target_is_directory=True)
        (root / ".zig-cache").symlink_to(newline)
        result = self.setup_cache(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different .zig-cache link", result.stderr)
        self.assertEqual(os.readlink(root / ".zig-cache"), str(newline))

    def test_concurrent_setups_leave_one_link_and_no_stray_ones(self):
        root = self.checkout("card")
        runs = [subprocess.Popen([root / "tools/zig/setup_cache.sh"], env=self.env) for _ in range(16)]
        self.assertEqual([run.wait(timeout=60) for run in runs], [0] * len(runs))
        self.assertEqual(os.readlink(root / ".zig-cache"), str(self.cache))
        self.assertEqual([path for path in self.cache.rglob("*") if path.is_symlink()], [])

    def test_concurrent_builds_from_two_checkouts_keep_their_own_outputs(self):
        env = self.stubs()
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
