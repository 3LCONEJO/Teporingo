def test_import_package():
    import teporingo_demultiplexing  # noqa: F401


def test_placeholder_math():
    assert 1 + 1 == 2


def test_load_default_config():
    from pathlib import Path

    from teporingo_demultiplexing.pipeline import run_pipeline

    config_path = Path("configs/default.yaml")
    summary = run_pipeline(config_path)

    assert summary["pipeline"]["min_maf"] == 0.01
    assert summary["pipeline"]["max_maf"] == 0.1
    assert summary["pipeline"]["use_gt"] is True


def test_load_quoted_config_values():
    from pathlib import Path

    from teporingo_demultiplexing.config import load_simple_yaml

    config = load_simple_yaml(Path("configs/test.yaml"))
    pipeline = config["pipeline"]

    assert pipeline["vcf"].startswith("/")
    assert pipeline["assignments"].startswith("/")
    assert pipeline["bams"]["batch_a"].startswith("/")


def test_run_pipeline_builds_metadata_and_batch_plan():
    from pathlib import Path

    from teporingo_demultiplexing.pipeline import run_pipeline

    summary = run_pipeline(Path("configs/test.yaml"))
    pipeline = summary["pipeline"]

    assert pipeline["vcf_metadata"]["sample_count"] > 0
    assert pipeline["batch_plan"][0]["batch_label"] == "batch_a"
    assert pipeline["batch_plan"][0]["bam_path"].endswith(".bam")
