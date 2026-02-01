import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from tqdm import tqdm


# Configuration
SCAN_DIRS: list[str] = []  # Add your SSD mount points here
OUTPUT_REPORT = "./backup_analysis_report.json"

# How many parent directories to include when matching duplicates
# e.g., 2 means "Documents/Photos/image.jpg" matches "Backup/Documents/Photos/image.jpg"
MATCH_PARENT_DIRS = 2

# Partial hash size for quick comparison (read first/last N bytes)
PARTIAL_HASH_SIZE = 8192  # 8KB from start + end


# ============ Path Classification Patterns ============

# Windows system/app paths (case-insensitive patterns)
WINDOWS_SYSTEM_PATTERNS = [
    "windows/",
    "program files/",
    "program files (x86)/",
    "programdata/",
    "appdata/local/",
    "appdata/locallow/",
    "appdata/roaming/microsoft/",
    "$recycle.bin/",
    "system volume information/",
    "recovery/",
    "msocache/",
    "windows.old/",
    "perflogs/",
    "config.msi/",
]

# Windows user paths (likely useful)
WINDOWS_USER_PATTERNS = [
    "documents/",
    "downloads/",
    "pictures/",
    "videos/",
    "music/",
    "desktop/",
    "onedrive/",
    "dropbox/",
]

# Android system/app paths
ANDROID_SYSTEM_PATTERNS = [
    "android/data/",
    "android/obb/",
    "android/media/",
    ".thumbnails/",
    ".cache/",
    "lost.dir/",
    "data/data/",
    "data/app/",
    "system/",
    "vendor/",
]

# Android user paths (likely useful)
ANDROID_USER_PATTERNS = [
    "dcim/",
    "download/",
    "pictures/",
    "movies/",
    "music/",
    "documents/",
    "whatsapp/media/",
    "telegram/",
]


@dataclass
class FileInfo:
    full_path: str
    filename: str
    size: int
    parent_dirs: str  # Last N parent directories
    category: str  # "user", "system", "unknown"
    partial_hash: str = ""


@dataclass
class DuplicateGroup:
    key: str  # filename + parent_dirs + size
    files: list[FileInfo] = field(default_factory=list)


def get_parent_dirs(path: str, num_dirs: int) -> str:
    """Extract the last N parent directories from a path."""
    parts = Path(path).parts
    if len(parts) <= num_dirs + 1:  # +1 for filename
        return "/".join(parts[:-1])
    return "/".join(parts[-(num_dirs + 1):-1])


def classify_path(path: str) -> str:
    """Classify a file path as 'user', 'system', or 'unknown'."""
    path_lower = path.lower().replace("\\", "/")

    # Check Windows patterns
    for pattern in WINDOWS_USER_PATTERNS:
        if pattern in path_lower:
            return "user"
    for pattern in WINDOWS_SYSTEM_PATTERNS:
        if pattern in path_lower:
            return "system"

    # Check Android patterns
    for pattern in ANDROID_USER_PATTERNS:
        if pattern in path_lower:
            return "user"
    for pattern in ANDROID_SYSTEM_PATTERNS:
        if pattern in path_lower:
            return "system"

    return "unknown"


def compute_partial_hash(filepath: str, size: int = PARTIAL_HASH_SIZE) -> str:
    """Compute a hash of the first and last N bytes of a file."""
    try:
        file_size = os.path.getsize(filepath)
        hasher = hashlib.md5()

        with open(filepath, "rb") as f:
            # Read from start
            hasher.update(f.read(size))

            # Read from end if file is large enough
            if file_size > size * 2:
                f.seek(-size, 2)
                hasher.update(f.read(size))

        return hasher.hexdigest()
    except (IOError, OSError):
        return ""


def scan_directory(root_dir: str) -> list[FileInfo]:
    """Scan a directory and collect file information."""
    files = []
    print(f"Scanning: {root_dir}")

    # First, count files for progress bar
    file_count = sum(1 for _ in Path(root_dir).rglob("*") if _.is_file())
    print(f"Found {file_count:,} files")

    for filepath in tqdm(Path(root_dir).rglob("*"), total=file_count, desc="Scanning"):
        if not filepath.is_file():
            continue

        try:
            full_path = str(filepath)
            file_info = FileInfo(
                full_path=full_path,
                filename=filepath.name,
                size=filepath.stat().st_size,
                parent_dirs=get_parent_dirs(full_path, MATCH_PARENT_DIRS),
                category=classify_path(full_path),
            )
            files.append(file_info)
        except (OSError, PermissionError):
            continue

    return files


def find_duplicates(files: list[FileInfo]) -> list[DuplicateGroup]:
    """Find duplicate files based on name + parent dirs + size."""
    # Group by key
    groups: dict[str, list[FileInfo]] = defaultdict(list)

    for f in files:
        key = f"{f.filename}|{f.parent_dirs}|{f.size}"
        groups[key].append(f)

    # Filter to only groups with duplicates
    duplicates = []
    for key, file_list in groups.items():
        if len(file_list) > 1:
            duplicates.append(DuplicateGroup(key=key, files=file_list))

    return duplicates


def verify_duplicates_with_hash(duplicates: list[DuplicateGroup]) -> list[DuplicateGroup]:
    """Verify duplicates by computing partial hashes."""
    print("\nVerifying duplicates with partial hashes...")
    verified = []

    for group in tqdm(duplicates, desc="Hashing"):
        # Compute hashes for all files in group
        hash_groups: dict[str, list[FileInfo]] = defaultdict(list)
        for f in group.files:
            f.partial_hash = compute_partial_hash(f.full_path)
            if f.partial_hash:
                hash_groups[f.partial_hash].append(f)

        # Keep only groups where hashes match
        for hash_val, file_list in hash_groups.items():
            if len(file_list) > 1:
                verified.append(DuplicateGroup(
                    key=f"{group.key}|{hash_val[:8]}",
                    files=file_list
                ))

    return verified


def generate_report(
    files: list[FileInfo],
    duplicates: list[DuplicateGroup],
    output_path: str
):
    """Generate a JSON report of the analysis."""
    # Category summary
    category_counts = {"user": 0, "system": 0, "unknown": 0}
    category_sizes = {"user": 0, "system": 0, "unknown": 0}

    for f in files:
        category_counts[f.category] += 1
        category_sizes[f.category] += f.size

    # Duplicate summary
    total_duplicate_files = sum(len(g.files) for g in duplicates)
    wasted_space = sum(
        sum(f.size for f in g.files[1:])  # All but one copy is "wasted"
        for g in duplicates
    )

    report = {
        "summary": {
            "total_files": len(files),
            "total_size_gb": round(sum(f.size for f in files) / (1024**3), 2),
            "duplicate_groups": len(duplicates),
            "duplicate_files": total_duplicate_files,
            "wasted_space_gb": round(wasted_space / (1024**3), 2),
        },
        "by_category": {
            cat: {
                "count": category_counts[cat],
                "size_gb": round(category_sizes[cat] / (1024**3), 2),
            }
            for cat in ["user", "system", "unknown"]
        },
        "duplicates": [
            {
                "filename": g.files[0].filename,
                "size_mb": round(g.files[0].size / (1024**2), 2),
                "copies": len(g.files),
                "paths": [f.full_path for f in g.files],
            }
            for g in sorted(duplicates, key=lambda x: -x.files[0].size)[:100]  # Top 100 by size
        ],
        "system_files_sample": [
            f.full_path for f in files if f.category == "system"
        ][:50],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_summary(report: dict):
    """Print a human-readable summary."""
    s = report["summary"]
    cat = report["by_category"]

    print("\n" + "=" * 50)
    print("BACKUP ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"\nTotal files: {s['total_files']:,}")
    print(f"Total size: {s['total_size_gb']:.2f} GB")

    print("\n--- By Category ---")
    for name, data in cat.items():
        print(f"  {name.upper():8} {data['count']:>10,} files  ({data['size_gb']:.2f} GB)")

    print("\n--- Duplicates ---")
    print(f"  Duplicate groups: {s['duplicate_groups']:,}")
    print(f"  Total duplicate files: {s['duplicate_files']:,}")
    print(f"  Wasted space: {s['wasted_space_gb']:.2f} GB")

    if report["duplicates"]:
        print("\n--- Top 10 Largest Duplicates ---")
        for dup in report["duplicates"][:10]:
            print(f"  {dup['filename']} ({dup['size_mb']:.1f} MB) - {dup['copies']} copies")


def main():
    if not SCAN_DIRS:
        print("ERROR: No directories configured to scan.")
        print("Edit SCAN_DIRS at the top of this file to add your SSD mount points.")
        print("\nExample:")
        print('  SCAN_DIRS = ["/Volumes/Backup1", "/Volumes/Backup2"]')
        sys.exit(1)

    # Collect all files
    all_files: list[FileInfo] = []
    for scan_dir in SCAN_DIRS:
        if os.path.isdir(scan_dir):
            all_files.extend(scan_directory(scan_dir))
        else:
            print(f"Warning: {scan_dir} is not a valid directory, skipping.")

    if not all_files:
        print("No files found to analyze.")
        sys.exit(1)

    # Find duplicates
    print(f"\nAnalyzing {len(all_files):,} files for duplicates...")
    duplicates = find_duplicates(all_files)
    print(f"Found {len(duplicates):,} potential duplicate groups")

    # Verify with hashes
    if duplicates:
        duplicates = verify_duplicates_with_hash(duplicates)
        print(f"Verified {len(duplicates):,} duplicate groups")

    # Generate report
    report = generate_report(all_files, duplicates, OUTPUT_REPORT)
    print(f"\nFull report saved to: {OUTPUT_REPORT}")

    # Print summary
    print_summary(report)


if __name__ == "__main__":
    main()
