from jinja2 import Environment, FileSystemLoader
from docx import Document
import os

def generate_act(data, template_path='templates/act_template.jinja2', output_path='act.docx'):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)
    rendered = template.render(data)
    # Сохраняем как docx (можно через python-docx)
    doc = Document()
    doc.add_paragraph(rendered)  # Упрощённо, лучше парсить HTML
    doc.save(output_path)