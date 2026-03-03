from __future__ import annotations

import argparse
from pathlib import Path

from src.ai_agent_course.orchestrator import EngineeringSurveyAgent
from src.ai_agent_course.secrets_loader import load_local_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Agent V2: анализ изображения и текстового запроса по ТЗ."
    )
    parser.add_argument("--image", required=True, help="Путь к изображению (магнитная/геофизическая карта)")
    parser.add_argument("--prompt", required=True, help="Инженерный запрос для агента")
    parser.add_argument("--miis-xml", default=None, help="Опциональный MIIS XML с учтёнными коммуникациями")
    parser.add_argument("--output-dir", default="output_v2", help="Каталог для результатов")
    parser.add_argument(
        "--llm-provider",
        default="auto",
        choices=["auto", "anthropic", "openai_compatible", "cursor"],
        help="Провайдер LLM",
    )
    parser.add_argument("--llm-api-key", default=None, help="API ключ LLM (опционально)")
    parser.add_argument("--llm-model", default=None, help="Имя модели LLM (опционально)")
    parser.add_argument("--llm-base-url", default=None, help="Base URL для openai_compatible")
    return parser


def main() -> None:
    load_local_env(Path(__file__).resolve().parent / ".env.local")
    args = build_parser().parse_args()
    agent = EngineeringSurveyAgent()
    result = agent.run(
        image_path=Path(args.image),
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
        miis_xml_path=Path(args.miis_xml) if args.miis_xml else None,
        llm_provider=args.llm_provider,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
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

