import os
import subprocess
import argparse
import sys

def format_files(directory, extensions, clang_format_path="clang-format"):
    """
    Recursively formats files with given extensions in the directory using clang-format.
    """
    # Normalize extensions to ensure they start with dot
    extensions = [ext if ext.startswith(".") else "." + ext for ext in extensions]
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    files_formatted = 0
    
    # Verify clang-format is available
    try:
        subprocess.run([clang_format_path, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print(f"Error: '{clang_format_path}' executable not found.")
        print("Please install clang-format or ensure it is in your system PATH.")
        print("Installation tips:")
        print("  Windows: winget install LLVM.LLVM")
        print("  Ubuntu/Debian: sudo apt install clang-format")
        print("  MacOS: brew install clang-format")
        return
    except Exception as e:
        print(f"Error checking clang-format: {e}")
        return

    print(f"Scanning '{directory}' for files with extensions: {', '.join(extensions)}...")

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                file_path = os.path.join(root, file)
                print(f"Formatting: {file_path}")
                try:
                    subprocess.run([clang_format_path, "-i", "-style=file", file_path], check=True)
                    files_formatted += 1
                except subprocess.CalledProcessError as e:
                    print(f"Failed to format {file_path}: {e}")

    print(f"Done. Formatted {files_formatted} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format code files using clang-format.")
    parser.add_argument("directory", help="The directory to scan for files (e.g. '.', 'c', 'h').")
    parser.add_argument("--extensions", "-e", nargs="+", default=[".c", ".h", ".cpp", ".hpp"], help="File extensions to format (default: .c .h .cpp .hpp)")
    parser.add_argument("--executable", default="clang-format", help="Path to clang-format executable (default: clang-format)")

    args = parser.parse_args()
    
    format_files(args.directory, args.extensions, args.executable)
