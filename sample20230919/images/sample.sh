find /home/ubuntu/codes/MuRAG/sample20230919/images/val -type f | shuf -n 5 | while read file; do
    # 使用 cp 命令将文件复制到新目录中
        cp "$file" /home/ubuntu/codes/MuRAG/sample20230919/images/test
done