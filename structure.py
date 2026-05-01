import os

def print_tree(start_path, indent=""):
    items = sorted(os.listdir(start_path))
    
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = i == len(items) - 1
        
        if is_last:
            prefix = "└── "
            next_indent = indent + "    "
        else:
            prefix = "├── "
            next_indent = indent + "│   "
        
        print(indent + prefix + item)
        
        if os.path.isdir(path):
            print_tree(path, next_indent)

if __name__ == "__main__":
    project_path = "ladybug"  # change this if needed
    print(project_path)
    print_tree(project_path)