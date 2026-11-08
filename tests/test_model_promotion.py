
from unittest.mock import patch

import src.manage_model as manager


def test_promotion_blocks_equal_accuracy(
    tmp_path, monkeypatch, test_checkpoint
):
    production_path = tmp_path / "best_model.pth"
    candidate_path = tmp_path / "candidate_model.pth"

    checkpoint_bytes = test_checkpoint.read_bytes()

    production_path.write_bytes(checkpoint_bytes)
    candidate_path.write_bytes(checkpoint_bytes)

    monkeypatch.setattr(
        manager, "PRODUCTION_PATH", production_path
    )
    monkeypatch.setattr(
        manager, "CANDIDATE_PATH", candidate_path
    )

    original_hash = manager.fingerprint(production_path)

    with (
        patch(
            "src.manage_model.create_dataloaders",
            return_value=(None, object(), None),
        ),
        patch(
            "src.manage_model.evaluate_checkpoint",
            side_effect=[0.98, 0.98],
        ),
        patch(
            "src.manage_model.deploy_candidate"
        ) as deploy,
        patch(
            "builtins.input",
            side_effect=AssertionError(
                "Confirmation must not be requested"
            ),
        ),
    ):
        manager.check_promotion()

    deploy.assert_not_called()

    assert manager.fingerprint(production_path) == original_hash
