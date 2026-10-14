"""Independent physical and chronological acceptance of reference artifacts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def validate_reference_artifacts(run_id, *, runs_root=None, metadata_root=None,
                                 progress=None, check_cancel=None):
    import pyarrow.parquet as pq
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services import reference_training as training
    from my_strategy.services.reference_model import load_model, SCHEMA_HASH
    root, report = training.load_reference_run(run_id, runs_root=runs_root, metadata_root=metadata_root,allow_validating=True)
    runs = root.parent
    plan = training._read(root / "config/training_plan.json")
    if plan["plan_hash"] != stable_hash({k:v for k,v in plan.items() if k != "plan_hash"}):
        raise ValueError("冻结计划哈希改变")
    if report["plan_id"] != plan["plan_id"] or report["source_reuse"] != plan["source_reuse"]:
        raise ValueError("模型报告与冻结来源/计划不一致")
    for name, digest in plan["code_hashes"].items():
        if training._sha(root / "config/frozen_code" / name) != digest:
            raise ValueError("原冻结代码改变：" + name)
    # Numeric code is required to understand stored tensors and labels. Runtime
    # or acceptance fixes have separate identity and cannot change the weights.
    if training._sha(Path(__file__).parent / "reference_model.py") != plan["code_hashes"]["reference_model.py"]:
        raise ValueError("数值模型实现与冻结代码不同")
    native, _, proof = training._source_header(plan["parameters"]["source_native_run_id"], runs)
    if proof != report["source_reuse"]:
        raise ValueError("来源证明改变")
    manifest_path = root / "dataset/manifest.json"
    if training._sha(manifest_path) != report["dataset"]["manifest_sha256"]:
        raise ValueError("新参考数据清单改变")
    manifest = training._read(manifest_path)
    records = manifest["records"]
    if (manifest["family"] != training.FAMILY or manifest["source_reuse"] != proof
            or manifest["code_hash"] != plan["code_hashes"]["reference_model.py"]
            or [r["symbol"] for r in records] != plan["cohort"]
            or report["dataset"]["stocks"] != len(records)):
        raise ValueError("参考数据身份或完整人口不一致")
    source_records = {r["symbol"]:r for r in native["dataset_records"]}

    def verify_record(record):
        if check_cancel: check_cancel()
        path = root / "dataset" / Path(record["path"]).name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("参考数据文件越界")
        if training._sha(path) != record["sha256"] or pq.read_metadata(path).num_rows != record["rows"]:
            raise ValueError("参考数据文件哈希或行数改变：" + record["symbol"])
        source = source_records[record["symbol"]]
        if record["source_sha256"] != source["sha256"]:
            raise ValueError("逐股来源特征身份改变")
        training._verified_record(source, runs)
        return record["rows"], record["eligible"], record["labels"]

    totals = [0, 0, 0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, values in enumerate(pool.map(verify_record, records), 1):
            totals = [a+b for a,b in zip(totals, values)]
            if progress and (index % 100 == 0 or index == len(records)):
                progress("reference_asset_validation", index, len(records))
    for key, actual in zip(("rows", "eligible", "labeled"), totals):
        if report["dataset"][key] != actual:
            raise ValueError("参考数据覆盖计数不符：" + key)
    folds = [*report["folds"], report["production"]]
    if [f["name"] for f in folds] != [s["name"] for s in plan["windows"]]:
        raise ValueError("参考训练窗口不完整")
    checks = []
    for index, (fold, split) in enumerate(zip(folds, plan["windows"]), 1):
        if check_cancel: check_cancel()
        session = load_model(fold["model_path"], device="cpu")
        model_manifest = training._read(Path(fold["model_path"]) / "manifest.json")
        if (model_manifest["schema_hash"] != SCHEMA_HASH or model_manifest["split"] != split
                or fold["split"] != split or model_manifest["validation"] != fold["validation"]
                or fold["predictive_gate"] != training._predictive_gate(fold["validation"])):
            raise ValueError("模型窗口/验证/预测门槛报告与资产不符：" + fold["name"])
        counts = model_manifest["validation"]["counts"]
        if any(sum(counts[name]) == 0 for name in ("train", "calibration", "policy")):
            raise ValueError("必需时间段没有成熟标签")
        if split["test_start"] and sum(counts["test"]) == 0:
            raise ValueError("历史留出窗口没有成熟标签")
        checks.append(dict(name=fold["name"], asset_valid=True,
            manifest_sha256=training._sha(Path(fold["model_path"]) / "manifest.json"),
            weights_sha256=model_manifest["weights_sha256"], predictive_gate=fold["predictive_gate"], counts=counts))
        del session
        if progress: progress("reference_model_validation", index, len(folds))
    result = dict(run_id=run_id, family=training.FAMILY, status="passed", technical_validation=True,
        source_assets_verified=len(records)*3, dataset_assets_verified=len(records),
        full_population=True, dataset=report["dataset"], folds=checks,
        training_started=False, weights_unchanged=True, publication_allowed=False,
        validation_code_sha256=training._sha(__file__))
    training._save(root / "validation/reference_validation.json", result)
    return result
