import os
import shutil

def define_cleanup_criteria(project_dir):
    unnecessary = [
        'audio/',  # Old downloads
        '__pycache__/',  # Cache
        '*.pyc',  # Compiled Python files
        'node_modules/',  # Frontend dependencies (will be reinstalled)
        '.DS_Store',  # macOS system files
        'venv/',  # Virtual environment (can be recreated)
        'models/fine_tuned_llama/checkpoint-*/',  # Keep only final checkpoint for fine-tuned Llama
        'transcripts/*_transcript.txt',  # Raw text transcripts (keep JSON)
        'tests/test_old*',  # Old irrelevant tests
    ]
    keep = ['cookies.txt', 'requirements.txt', 'README.md', 'docs/', 'src/', 'frontend/src/', 'frontend/package.json']

    # Keep only the latest checkpoints for fine-tuned Llama model
    unnecessary.append('models/fine_tuned_llama/checkpoint-100/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-125/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-150/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-175/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-200/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-225/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-250/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-275/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-300/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-325/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-350/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-375/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-400/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-425/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-450/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-475/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-50/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-525/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-550/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-575/')
    unnecessary.append('models/fine_tuned_llama/checkpoint-75/')

    return unnecessary, keep

def clean_project(project_dir, unnecessary, keep):
    """Clean the project by removing unnecessary files while preserving essential ones."""
    removed_count = 0

    for root, dirs, files in os.walk(project_dir, topdown=False):
        for name in files + dirs:
            path = os.path.join(root, name)

            # Skip if in keep list
            if any(k in path for k in keep):
                continue

            # Check if matches unnecessary patterns
            should_remove = False
            for u in unnecessary:
                if u.endswith('/') and u.rstrip('/') in path:
                    should_remove = True
                    break
                elif not u.endswith('/') and name == u:
                    should_remove = True
                    break
                elif u.startswith('*.') and name.endswith(u[1:]):
                    should_remove = True
                    break

            if should_remove:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"Removed file: {path}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Removed directory: {path}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {path}: {e}")

    print(f"Cleanup complete. Removed {removed_count} items.")
    return removed_count

if __name__ == "__main__":
    project_dir = "/Users/RRL_1/riffter"
    unnecessary, keep = define_cleanup_criteria(project_dir)

    # Validation
    assert 'cookies.txt' in keep, "cookies.txt must be preserved"
    assert len(unnecessary) > 5, "Should have multiple cleanup targets"

    print("Cleanup criteria defined:")
    print(f"Unnecessary: {unnecessary}")
    print(f"Keep: {keep}")

    # Actually perform cleanup
    removed = clean_project(project_dir, unnecessary, keep)
    print(f"Removed {removed} items")
