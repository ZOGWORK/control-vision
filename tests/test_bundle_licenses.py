import importlib.metadata
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bundle_licenses  # noqa: E402


class LicenseBundleTests(unittest.TestCase):
    def test_top_level_names_skip_pyinstaller_internals(self):
        names = bundle_licenses.top_level_names(
            ["yaml.constructor", "cv2/cv2.pyd", "PySide6\\Qt6Core.dll", "pyi_rth_inspect", "pyimod02_importers",
             "base_library.zip"])
        self.assertEqual(names, {"yaml", "cv2", "PySide6"})

    def test_bundle_copies_license_files_and_indexes_each_distribution(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "python-packages"
            out.mkdir()
            (out / "stale.txt").write_text("old build", encoding="utf-8")
            pairs = bundle_licenses.write_license_bundle(["yaml", "numpy.core"], out, always=["pyinstaller"])
            index = (out / bundle_licenses.INDEX_NAME).read_text(encoding="utf-8")
            pyyaml = f"pyyaml-{importlib.metadata.version('PyYAML')}"
            self.assertFalse((out / "stale.txt").exists())
            self.assertIn(f"PyYAML {importlib.metadata.version('PyYAML')}: MIT", index)
            self.assertRegex(index, r"(?m)^numpy \S+: ")
            self.assertRegex(index, r"(?mi)^pyinstaller \S+: ")
            self.assertTrue(any(path.is_file() and path.name.upper().startswith("LICENSE")
                                for path in (out / pyyaml).rglob("*")))
            self.assertIn((str(out / bundle_licenses.INDEX_NAME), "."), pairs)
            self.assertTrue(all(Path(source).is_file() for source, _ in pairs))
            self.assertTrue(any(folder.split("/")[0] == pyyaml for _, folder in pairs))

    def test_distribution_without_license_file_is_listed_with_declared_license(self):
        dist = importlib.metadata.distribution("rapidocr-onnxruntime")
        self.assertEqual(bundle_licenses.license_files(dist), [])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "python-packages"
            bundle_licenses.write_license_bundle(["rapidocr_onnxruntime"], out)
            index = (out / bundle_licenses.INDEX_NAME).read_text(encoding="utf-8")
        self.assertIn(f"{dist.metadata['Name']} {dist.version}: Apache-2.0", index)
        self.assertIn("No license file is shipped in this package", index)


if __name__ == "__main__":
    unittest.main()
