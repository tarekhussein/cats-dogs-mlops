
import argparse
import hashlib
from pathlib import Path
from src.compare_models import evaluate_checkpoint
from src.dataset import create_dataloaders
import torch
import os
import shutil
from datetime import datetime, timezone

PRODUCTION_PATH = Path("models/best_model.pth")
CANDIDATE_PATH = Path("models/candidate_model.pth")
BACKUP_DIR = Path("models/backups")


def fingerprint(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def inspect_checkpoint(name, path):
    print(f"\n{name}")
    print("-" * 45)

    if not path.is_file():
        print("Status: NOT FOUND")
        return

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    print("Path:", path)
    print("SHA-256:", fingerprint(path))
    print("Epoch:", checkpoint.get("epoch", "N/A"))

    accuracy = checkpoint.get("val_accuracy")
    if accuracy is not None:
        print(f"Saved validation accuracy: {accuracy:.2%}")


def show_status():
    print("MODEL MANAGEMENT STATUS")
    print("=" * 45)

    inspect_checkpoint("Production model", PRODUCTION_PATH)
    inspect_checkpoint("Candidate model", CANDIDATE_PATH)

    print("\nAvailable backups:")
    if BACKUP_DIR.exists():
        backups = sorted(BACKUP_DIR.glob("*.pth"))
        for backup in backups:
            print(" -", backup.name)
        if not backups:
            print(" None")
    else:
        print(" None")


def deploy_candidate():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Save the current production checkpoint first.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    backup_path = BACKUP_DIR / f"production_{timestamp}.pth"

    shutil.copy2(PRODUCTION_PATH, backup_path)

    # Verify that the backup is identical.
    original_hash = fingerprint(PRODUCTION_PATH)
    backup_hash = fingerprint(backup_path)

    if original_hash != backup_hash:
        raise RuntimeError("Backup verification failed")

    print("Backup created:", backup_path)

    # Stage the candidate in the production directory.
    temporary_path = PRODUCTION_PATH.with_name(
        PRODUCTION_PATH.name + ".tmp"
    )

    try:
        shutil.copy2(CANDIDATE_PATH, temporary_path)

        if fingerprint(temporary_path) != fingerprint(CANDIDATE_PATH):
            raise RuntimeError("Candidate copy verification failed")

        # Atomic replacement on the same filesystem.
        os.replace(temporary_path, PRODUCTION_PATH)

    finally:
        temporary_path.unlink(missing_ok=True)

    print("Production model updated successfully.")
    print("New model SHA-256:", fingerprint(PRODUCTION_PATH))


def check_promotion():
    if not PRODUCTION_PATH.is_file():
        raise FileNotFoundError(PRODUCTION_PATH)

    if not CANDIDATE_PATH.is_file():
        raise FileNotFoundError(CANDIDATE_PATH)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Running fresh validation comparison...")
    print("Device:", device)

    _, val_loader, _ = create_dataloaders()

    production_accuracy = evaluate_checkpoint(
        PRODUCTION_PATH, val_loader, device
    )

    candidate_accuracy = evaluate_checkpoint(
        CANDIDATE_PATH, val_loader, device
    )

    print(f"Production accuracy: {production_accuracy:.2%}")
    print(f"Candidate accuracy:  {candidate_accuracy:.2%}")

    if candidate_accuracy <= production_accuracy:
        print("PROMOTION BLOCKED: Candidate did not improve accuracy.")
        return

    print("Candidate passed the validation gate.")

    confirmation = input(
        "Type PROMOTE to authorize deployment: "
    )

    if confirmation != "PROMOTE":
        print("Promotion cancelled.")
        return

    print("Promotion authorized.")
    deploy_candidate()


def rollback_model(backup_name):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    backup_path = BACKUP_DIR / backup_name

    if (
        backup_path.parent.resolve() != BACKUP_DIR.resolve()
        or backup_path.suffix != ".pth"
        or not backup_path.is_file()
    ):
        raise ValueError("Invalid or missing backup checkpoint")

    checkpoint = torch.load(
        backup_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        "model_state_dict" not in checkpoint
        or checkpoint.get("class_to_idx") != {"Cat": 0, "Dog": 1}
    ):
        raise ValueError("Invalid backup checkpoint contents")

    print("Selected backup:", backup_path)
    print("Backup SHA-256:", fingerprint(backup_path))

    confirmation = input(
        "Type ROLLBACK to restore this checkpoint: "
    )

    if confirmation != "ROLLBACK":
        print("Rollback cancelled.")
        return

    # Preserve the current production model before restoration.
    if PRODUCTION_PATH.is_file():
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        safety_backup = BACKUP_DIR / f"pre_rollback_{timestamp}.pth"
        shutil.copy2(PRODUCTION_PATH, safety_backup)

        if fingerprint(safety_backup) != fingerprint(PRODUCTION_PATH):
            raise RuntimeError("Safety backup verification failed")

        print("Current production backed up:", safety_backup)

    temporary_path = PRODUCTION_PATH.with_name(
        PRODUCTION_PATH.name + ".rollback.tmp"
    )

    try:
        shutil.copy2(backup_path, temporary_path)

        if fingerprint(temporary_path) != fingerprint(backup_path):
            raise RuntimeError("Rollback copy verification failed")

        os.replace(temporary_path, PRODUCTION_PATH)

    finally:
        temporary_path.unlink(missing_ok=True)

    print("Rollback completed successfully.")
    print("Restored SHA-256:", fingerprint(PRODUCTION_PATH))

def main():
    parser = argparse.ArgumentParser(
        description="Cats vs Dogs model management"
    )

    parser.add_argument(
        "command",
        choices=["status", "promote", "rollback"],
    )

    parser.add_argument(
        "--backup",
        help="Backup filename to restore",
    )
    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "promote":
        check_promotion()
    elif args.command == "rollback":
        if not args.backup:
            parser.error("rollback requires --backup")

        rollback_model(args.backup)
    


if __name__ == "__main__":
    main()
