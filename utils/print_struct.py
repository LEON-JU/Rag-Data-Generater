import os
import sys
import stat

def get_file_size(file_path):
    """获取文件大小，返回格式化字符串"""
    try:
        if os.path.isdir(file_path):
            return '[目录]'
        else:
            size = os.path.getsize(file_path)
            # 格式化文件大小
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f'{size:.1f} {unit}'
                size /= 1024.0
            return f'{size:.1f} TB'
    except (OSError, PermissionError):
        return '[无法访问]'

def print_tree(path, prefix='', level=0, max_per_level=20):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        print(prefix + ' [权限不足]')
        return

    # 只有 level >= 1 时才限制数量
    if level >= 1 and len(entries) > max_per_level:
        display_entries = entries[:max_per_level]
        truncated = True
    else:
        display_entries = entries
        truncated = False

    for i, name in enumerate(display_entries):
        full_path = os.path.join(path, name)
        is_last = (i == len(display_entries) - 1)
        if is_last and not truncated:
            connector = '└── '
            next_prefix = prefix + '    '
        else:
            connector = '├── '
            next_prefix = prefix + '│   '

        # 获取文件大小并打印
        size_info = get_file_size(full_path)
        print(f'{prefix}{connector}{name} ({size_info})')

        if os.path.isdir(full_path):
            print_tree(full_path, next_prefix, level + 1, max_per_level)

    if truncated:
        print(prefix + ('└── ' if not display_entries else '├── ') + '...')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python3 print_struct.py <目录路径>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print("错误：路径不是一个目录")
        sys.exit(1)

    print(f"{root} [目录]")
    print_tree(root)