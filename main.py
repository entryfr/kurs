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
        "--construction-zones",
        default=None,
        help="Опциональный путь к GeoPackage с зонами строительства.",
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
    parser.add_argument(
        "--miis-xml",
        default=None,
        help="Опциональный путь к MIIS XML. Если указан, аномалии/коммуникации берутся из XML.",
    )
    parser.add_argument(
        "--segy",
        default=None,
        help="Опциональный путь к SEG-Y для извлечения дополнительных признаков.",
    )
    parser.add_argument(
        "--norms-profile",
        default="normative",
        choices=["demo", "normative", "strict"],
        help="Профиль нормативных допусков.",
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
        miis_xml_path=Path(args.miis_xml) if args.miis_xml else None,
        segy_path=Path(args.segy) if args.segy else None,
        construction_zones_path=Path(args.construction_zones) if args.construction_zones else None,
        norms_profile=args.norms_profile,
    )

    logger.info("Пайплайн завершён: %s", summary)
    print("Готово. Выходные файлы:")
    print(f"  - {summary['dxf_path']}")
    print(f"  - {summary['xml_path']}")
    print(f"  - {summary['act_path']}")
    if summary.get("act_pdf_path"):
        print(f"  - {summary['act_pdf_path']}")
    print(f"  - Режим входа: {summary['input_mode']}")
    print(f"  - Профиль норм: {summary['norms_profile']}")


if __name__ == "__main__":
    main()