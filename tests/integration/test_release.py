from scripts import audit_environment, verify_release


def scan(tmp_path, files):
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    report = verify_release.Report()
    verify_release.scan_repository(tmp_path, list(files), report)
    return report.failures


def test_scan_accepts_clean_repository(tmp_path):
    assert scan(tmp_path, {"src/a.py": "x = '/path/to/chest_xray'\n",
                           "report/figures/roc.png": b"\x89PNG", ".env.example": "XRAY_DATA_ROOT=\n"}) == []


def test_scan_flags_secrets_paths_and_forbidden_files(tmp_path):
    failures = scan(tmp_path, {
        ".env": "XRAY_DATA_ROOT=x\n",
        "kaggle.json": "{}",
        "outputs/best.pt": b"\0",
        "data/raw/x.jpeg": b"\xff\xd8",
        "src/leak.png": b"\x89PNG",
        "notes.md": "PrivateKey = " + "A" * 43 + "=\n",
        "cfg.py": "root = 'C:" + "\\\\Users\\\\someone\\\\data'\n",
        "run.sh": "cd /home/" + "member/xray\n",
        "token.txt": "http://jupyter?token=" + "a" * 48 + "\n",
    })
    text = "\n".join(failures)
    for expected in (".env", "kaggle.json", ".pt", ".jpeg", "src/leak.png", "WireGuard",
                     "Windows user path", "Linux home path", "Jupyter token"):
        assert expected in text, expected


def test_release_verification_passes_on_repository():
    assert verify_release.main(["--skip-inference"]) == 0


def test_lock_file_pins_every_requirement(tmp_path):
    names = audit_environment.requirement_names()
    assert {"torch", "torchvision", "pandas", "PyYAML"} <= set(names)
    versions = audit_environment.installed_versions(names)
    text = audit_environment.lock_text(versions, {"cuda_runtime": None})
    pins = [line for line in text.splitlines() if not line.startswith("#")]
    assert pins == [f"{name}=={versions[name]}" for name in names]
