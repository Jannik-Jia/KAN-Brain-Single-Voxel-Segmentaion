#!/usr/bin/env python
# coding: utf-8
"""
Balanced 数据快速一致性检查（仅做 1 和 5）
1) 文件存在性 + MD5
5) 跨受试者重复文件检测
"""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import pandas as pd
from tqdm import tqdm

def compute_file_hash(file_path, chunk_size=1024 * 1024):
    """计算文件的 MD5 哈希（按块读取，避免占内存）"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            md5.update(buf)
    return md5.hexdigest()

def check_subject_files(subject_dir: Path):
    """
    仅检查：balanced_output 是否存在；4D/3D 文件是否存在；计算 MD5。
    返回最小信息给后续去做跨受试者重复检测。
    """
    balanced_dir = subject_dir / "balanced_output"
    data_4d = balanced_dir / "balanced_data_4d10000.nii.gz"
    label_3d = balanced_dir / "balanced_labels_3d10000.nii.gz"

    result = {
        "subject_id": subject_dir.name,
        "status": "unknown",
        "balanced_dir_exists": balanced_dir.exists(),
        "data_4d_exists": False,
        "label_3d_exists": False,
        "data_4d_hash": None,
        "label_3d_hash": None,
        "data_4d_path": str(data_4d),
        "label_3d_path": str(label_3d),
    }

    if not balanced_dir.exists():
        result["status"] = "no_balanced_dir"
        return result

    d_exists = data_4d.exists()
    l_exists = label_3d.exists()
    result["data_4d_exists"] = d_exists
    result["label_3d_exists"] = l_exists

    if not d_exists or not l_exists:
        result["status"] = "missing_files"
        return result

    # 计算哈希（只在文件存在时）
    try:
        result["data_4d_hash"] = compute_file_hash(data_4d)
        result["label_3d_hash"] = compute_file_hash(label_3d)
        result["status"] = "ok"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result

def find_duplicates(all_results):
    """
    跨受试者重复文件检测：
    - 4D 数据哈希重复
    - 3D 标签哈希重复
    返回结构里给出 hash -> subjects 列表
    """
    # 只统计 status 为 ok 的
    hash_4d_map = {}   # hash -> [subjects]
    hash_3d_map = {}

    for r in all_results:
        if r.get("status") != "ok":
            continue
        h4 = r.get("data_4d_hash")
        h3 = r.get("label_3d_hash")
        sid = r.get("subject_id")

        if h4:
            hash_4d_map.setdefault(h4, []).append(sid)
        if h3:
            hash_3d_map.setdefault(h3, []).append(sid)

    # 只挑出出现次数 > 1 的哈希
    dup_4d = {h: sids for h, sids in hash_4d_map.items() if len(sids) > 1}
    dup_3d = {h: sids for h, sids in hash_3d_map.items() if len(sids) > 1}

    # 汇总为便于展示的列表
    duplicate_list = []
    for h, sids in dup_4d.items():
        duplicate_list.append({
            "type": "4D_data",
            "hash": h,
            "subjects": sids
        })
    for h, sids in dup_3d.items():
        duplicate_list.append({
            "type": "3D_label",
            "hash": h,
            "subjects": sids
        })

    return duplicate_list, dup_4d, dup_3d

def main():
    ROOT_DIR = Path("/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS")
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Balanced 数据快速一致性检查（仅做 1 和 5）")
    print("=" * 80)
    print(f"数据目录: {ROOT_DIR}")
    print(f"输出目录: {output_dir}")

    subject_dirs = sorted([d for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("FOR_")])
    print(f"\n找到 {len(subject_dirs)} 个受试者")

    # 逐个受试者做文件存在性与 MD5
    all_results = []
    for d in tqdm(subject_dirs, desc="检查进度"):
        all_results.append(check_subject_files(d))

    # 状态统计
    status_counts = {}
    for r in all_results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    print("\n状态统计：")
    for k, v in status_counts.items():
        print(f"  {k}: {v} 个受试者")

    # 跨受试者重复
    duplicates, dup4_map, dup3_map = find_duplicates(all_results)

    if duplicates:
        print(f"\n❌ 发现 {len(duplicates)} 组重复文件：")
        for item in duplicates:
            h = item["hash"][:16] + "..."
            print(f"  - {item['type']} 重复：哈希 {h}  受试者 {', '.join(item['subjects'])}")
    else:
        print("\n✅ 未发现重复文件")

    # 保存 JSON 报告
    report = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(ROOT_DIR),
        "total_subjects": len(subject_dirs),
        "status_summary": status_counts,
        "duplicates": duplicates,
        "all_results": all_results,
    }
    json_path = output_dir / f"consistency_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON 报告已保存：{json_path}")

    # 保存 CSV 摘要（轻量字段）
    rows = []
    for r in all_results:
        rows.append({
            "subject_id": r["subject_id"],
            "status": r["status"],
            "balanced_dir_exists": r["balanced_dir_exists"],
            "data_4d_exists": r["data_4d_exists"],
            "label_3d_exists": r["label_3d_exists"],
            "data_4d_hash": (r["data_4d_hash"] or "")[:16],
            "label_3d_hash": (r["label_3d_hash"] or "")[:16],
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"consistency_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 CSV 摘要已保存：{csv_path}")

    # 操作建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    if status_counts.get("no_balanced_dir", 0) > 0:
        print("⚠️ 存在缺少 balanced_output 目录的受试者，请补齐目录或排查路径。")
    if status_counts.get("missing_files", 0) > 0:
        print("⚠️ 存在缺少 4D/3D 文件的受试者，请补齐文件。")
    if duplicates:
        print("⚠️ 检测到重复文件：请核对复制流程，替换为正确受试者对应的文件。")
    if not duplicates and status_counts.get("ok", 0) == len(subject_dirs):
        print("✅ 所有受试者文件存在且未发现重复，可以继续后续流程。")

    print("\n检查完成！")
    return report

if __name__ == "__main__":
    main()
