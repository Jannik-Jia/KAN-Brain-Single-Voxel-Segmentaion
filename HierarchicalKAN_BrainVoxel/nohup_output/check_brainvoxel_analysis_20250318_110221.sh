#!/bin/bash
# 查看和管理 brainvoxel_analysis_20250318_110221 任务的脚本

function check_status() {
  if ps -p 1983499 > /dev/null; then
    echo "✅ 任务正在运行 (PID: 1983499)"
  else
    echo "❌ 任务已停止运行"
    
    # 检查是否有结束时间记录
    if grep -q "分析完成" "./nohup_output/brainvoxel_analysis_20250318_110221.out"; then
      echo "✓ 任务已正常完成"
      grep "分析完成" "./nohup_output/brainvoxel_analysis_20250318_110221.out" -A 3
    else
      echo "! 任务可能异常终止，检查输出文件末尾:"
      tail -n 20 "./nohup_output/brainvoxel_analysis_20250318_110221.out"
    fi
  fi
}

function show_output() {
  tail -n ${1:-50} "./nohup_output/brainvoxel_analysis_20250318_110221.out"
}

function follow_output() {
  tail -f "./nohup_output/brainvoxel_analysis_20250318_110221.out"
}

case "$1" in
  status)
    check_status
    ;;
  output)
    show_output $2
    ;;
  follow)
    follow_output
    ;;
  kill)
    echo "正在停止任务 (PID: 1983499)..."
    kill $2 1983499
    ;;
  *)
    echo "使用方法: $0 {status|output [行数]|follow|kill [-9]}"
    echo "  status - 检查任务状态"
    echo "  output [行数] - 显示输出文件的最后N行 (默认50行)"
    echo "  follow - 实时跟踪输出"
    echo "  kill [-9] - 停止任务 (可选-9参数强制终止)"
    ;;
esac
