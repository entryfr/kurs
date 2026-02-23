import argparse
import logging
from pathlib import Path
from src.pipeline.service import run_pipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Демо-пайплайн анализа геофизических данных подземных коммуникаций."
    )
    parser.add_argument(
        "--magnetic-grid",
        default="data/raw/magnetic_grid.npy",
        help="Путь к файлу .npy с магнитной сеткой.",
    )
    parser.add_argument(
        "--anomalies",
        default="data/processed/anomalies.gpkg",
        help="Путь к GeoPackage с аномалиями.",
    )
    parser.add_argument(
        "--utilities",
        default="data/processed/utilities.gpkg",
        help="Путь к GeoPackage с учтёнными коммуникациями.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Каталог для выходных файлов DXF/XML/DOCX.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=30.0,
        help="Порог детекции аномалий в нТл.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args()
    summary = run_pipeline(
        magnetic_grid_path=Path(args.magnetic_grid),
        anomalies_path=Path(args.anomalies),
        utilities_path=Path(args.utilities),
        output_dir=Path(args.output_dir),
        anomaly_threshold=args.anomaly_threshold,
    )

    logger.info("Пайплайн завершён: %s", summary)
    print("Готово. Выходные файлы:")
    print(f"  - {summary['dxf_path']}")
    print(f"  - {summary['xml_path']}")
    print(f"  - {summary['act_path']}")


if __name__ == "__main__":
    main()