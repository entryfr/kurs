from __future__ import annotations

import argparse
from pathlib import Path

from src.ai_agent_course.orchestrator import EngineeringSurveyAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Agent V2: анализ изображения и текстового запроса по ТЗ."
    )
    parser.add_argument("--image", required=True, help="Путь к изображению (магнитная/геофизическая карта)")
    parser.add_argument("--prompt", required=True, help="Инженерный запрос для агента")
    parser.add_argument("--miis-xml", default=None, help="Опциональный MIIS XML с учтёнными коммуникациями")
    parser.add_argument("--output-dir", default="output_v2", help="Каталог для результатов")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent = EngineeringSurveyAgent()
    result = agent.run(
        image_path=Path(args.image),
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
        miis_xml_path=Path(args.miis_xml) if args.miis_xml else None,
    )

    print("=== AI Agent V2 ===")
    print("Mode:", result.mode)
    print("Answer:", result.answer)
    print("Artifacts:")
    print(" -", result.artifacts.dxf_path)
    print(" -", result.artifacts.miis_xml_path)
    print(" -", result.artifacts.pdf_report_path)


if __name__ == "__main__":
    main()

