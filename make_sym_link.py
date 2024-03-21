def create_symbolic_links(src_dir_path, tar_dir_path):
    import os

    # Function to create symbolic links for bmp and json files
    def create_links(src, dst):
        if os.path.isfile(src) and (src.endswith(".bmp") or src.endswith(".json")):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)

    # Walk through the source directory and create symbolic links in the target directory
    for root, dirs, files in os.walk(src_dir_path):
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, src_dir_path)
            dst_path = os.path.join(tar_dir_path, rel_path)
            create_links(src_path, dst_path)

    print("Symbolic links created successfully.")

# Example usage
src_dir_path = "/home/jay/mnt/hdd01/data/hankook_tire/raw_data/Line_Scan_pattern분류"
tar_dir_path = "/home/jay/mnt/hdd01/data/hankook_tire/symbolic/Line_Scan_pattern분류"
create_symbolic_links(src_dir_path, tar_dir_path)