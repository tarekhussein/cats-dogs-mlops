
import hashlib
from unittest.mock import patch

import src.manage_model as manager


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rollback_restores_previous_model(
    tmp_path, monkeypatch, test_checkpoint
):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    production_path = tmp_path / "best_model.pth"
    original_backup = backup_dir / "original.pth"

    # Use the existing valid checkpoint as test data.
    checkpoint_bytes = test_checkpoint.read_bytes()

    original_backup.write_bytes(checkpoint_bytes)

    # Simulate a changed production checkpoint.
    production_path.write_bytes(checkpoint_bytes + b"changed")

    previous_production_hash = sha256(production_path)
    expected_hash = sha256(original_backup)

    monkeypatch.setattr(
        manager, "PRODUCTION_PATH", production_path
    )
    monkeypatch.setattr(
        manager, "BACKUP_DIR", backup_dir
    )

    with patch("builtins.input", return_value="ROLLBACK"):
        manager.rollback_model("original.pth")

    # The original checkpoint must be restored.
    assert sha256(production_path) == expected_hash

    # The previous production checkpoint must be backed up.
    safety_backups = list(
        backup_dir.glob("pre_rollback_*.pth")
    )

    assert len(safety_backups) == 1

    assert (
        sha256(safety_backups[0])
        == previous_production_hash
    )
