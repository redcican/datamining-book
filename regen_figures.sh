#!/bin/bash
# 重新生成所有图片
PYTHON=/home/cican/miniconda3/bin/python3
PROJECT=/home/cican/paper/datamining-book

cd "$PROJECT"

SCRIPTS=$(find code -name "fig*.py" | sort)
TOTAL=$(echo "$SCRIPTS" | wc -l)
SUCCESS=0
FAILED=0
FAILED_LIST=""

echo "共 $TOTAL 个脚本，开始生成..."
echo ""

for script in $SCRIPTS; do
    ch=$(dirname "$script")
    name=$(basename "$script" .py)
    printf "  %-50s" "$script"
    output=$(PYTHONPATH="$PROJECT/code" $PYTHON "$script" 2>&1)
    if [ $? -eq 0 ]; then
        echo "✓"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "✗ FAILED"
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST\n  $script: $(echo "$output" | tail -1)"
    fi
done

echo ""
echo "完成：$SUCCESS 成功，$FAILED 失败"
if [ -n "$FAILED_LIST" ]; then
    echo -e "失败列表：$FAILED_LIST"
fi
echo ""
echo "图片总数：$(find public/figures -name '*.png' | wc -l)"
