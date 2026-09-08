#!/bin/bash

OUTPUT_FILE="yolop_full_code_dump.txt"
echo "# YOLOP PROJECT FULL CODE DUMP" > $OUTPUT_FILE
echo "# Generated on: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Find all files excluding certain directories and binary files
find . -type f \
    ! -path "*/.*" \
    ! -path "*/.venv/*" \
    ! -path "*/checkpoints/*" \
    ! -path "*/dataset/*" \
    ! -path "*/archive/*" \
    ! -path "*/logs/*" \
    ! -path "*/datasets/*" \
    ! -path "*/exports/*" \
    ! -path "*/tests/*" \
    ! -path "*/data/*" \
    ! -path "*/models/*" \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    ! -name "*.png" \
    ! -name "*.jpg" \
    ! -name "*.jpeg" \
    ! -name "*.json" \
    ! -name "*.zip" \
    ! -name "*.sh" \
    -print | while read -r file; do

    # Skip the dump file itself
    if [[ "$file" == "./$OUTPUT_FILE" ]]; then
        continue
    fi

    echo "================================================================================" >> $OUTPUT_FILE
    echo "FILE: $file" >> $OUTPUT_FILE
    echo "================================================================================" >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo -e "\n" >> $OUTPUT_FILE
done

echo "Done! All code files have been dumped into $OUTPUT_FILE"
