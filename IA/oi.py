"""Funções seguras e independentes da interface do Assistente."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import secrets
import statistics
import string
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
VISION_MODEL = "qwen2.5vl:7b"

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calculate_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _calculate_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool) or not math.isfinite(node.value):
            raise ValueError("Use apenas números finitos.")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("O expoente precisa estar entre -10 e 10.")

        result = _BINARY_OPERATORS[type(node.op)](left, right)

        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError("O resultado precisa ser um número real.")

        if not math.isfinite(float(result)) or abs(result) > 1e100:
            raise ValueError("O resultado é grande demais.")

        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_calculate_node(node.operand))

    raise ValueError("A conta contém uma operação que não é permitida.")


def safe_calculate(expression: str) -> float | int:
    """Calcula apenas operações aritméticas; nunca executa código Python."""
    expression = expression.strip().replace(",", ".")

    if not expression or len(expression) > 120:
        raise ValueError("Digite uma conta curta usando números e sinais.")

    try:
        tree = ast.parse(expression, mode="eval")
        return _calculate_node(tree)
    except (SyntaxError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("Não consegui calcular essa expressão.") from error


def try_calculate_message(message: str) -> str | None:
    """Reconhece perguntas simples de matemática em português."""
    candidate = message.strip().lower()

    candidate = re.sub(
        r"^(?:quanto\s+(?:é|e|dá|da)|calcule|calcula|resolva|qual\s+(?:é|e)\s*)\s*",
        "",
        candidate,
    )

    candidate = candidate.rstrip(" ?=.")
    candidate = candidate.replace("^", "**")

    if not re.fullmatch(r"[0-9+\-*/%().\s,]+", candidate):
        return None

    if not re.search(r"\d", candidate) or not re.search(r"[+\-*/%]", candidate):
        return None

    try:
        answer = safe_calculate(candidate)
    except ValueError:
        return None

    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)

    return f"O resultado é {answer}."


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("O resultado não é finito.")
    if abs(value - round(value)) < 1e-10:
        return str(round(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", ",")


def _parse_date(value: str) -> date:
    value = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value)
        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError as error:
        raise ValueError("Use uma data válida no formato DD/MM/AAAA ou AAAA-MM-DD.") from error


_UNIT_FACTORS: dict[str, tuple[str, float, str]] = {}


def _add_units(group: str, factor: float, display: str, *aliases: str) -> None:
    for alias in aliases:
        _UNIT_FACTORS[_normalize_text(alias)] = (group, factor, display)


_add_units("comprimento", 0.001, "mm", "mm", "milimetro", "milimetros")
_add_units("comprimento", 0.01, "cm", "cm", "centimetro", "centimetros")
_add_units("comprimento", 1, "m", "m", "metro", "metros")
_add_units("comprimento", 1000, "km", "km", "quilometro", "quilometros")
_add_units("comprimento", 0.0254, "pol", "in", "inch", "inches", "polegada", "polegadas")
_add_units("comprimento", 0.3048, "ft", "ft", "foot", "feet", "pe")
_add_units("comprimento", 1609.344, "mi", "mi", "mile", "miles", "milha", "milhas")
_add_units("massa", 0.001, "mg", "mg", "miligramas", "miligrama")
_add_units("massa", 1, "g", "g", "grama", "gramas")
_add_units("massa", 1000, "kg", "kg", "quilo", "quilos", "quilograma", "quilogramas")
_add_units("massa", 1_000_000, "t", "t", "tonelada", "toneladas")
_add_units("massa", 453.59237, "lb", "lb", "lbs", "libra", "libras")
_add_units("massa", 28.349523125, "oz", "oz", "ounce", "ounces", "onca", "oncas")
_add_units("volume", 0.001, "mL", "ml", "mililitro", "mililitros")
_add_units("volume", 1, "L", "l", "litro", "litros")
_add_units("volume", 1000, "m3", "m3", "metro cubico", "metros cubicos")
_add_units("dados", 1, "B", "b", "byte", "bytes")
_add_units("dados", 1_000, "kB", "kb", "kbyte", "kilobyte", "kilobytes")
_add_units("dados", 1_000_000, "MB", "mb", "megabyte", "megabytes")
_add_units("dados", 1_000_000_000, "GB", "gb", "gigabyte", "gigabytes")
_add_units("dados", 1_000_000_000_000, "TB", "tb", "terabyte", "terabytes")
_add_units("tempo", 1, "s", "s", "segundo", "segundos")
_add_units("tempo", 60, "min", "min", "minuto", "minutos")
_add_units("tempo", 3600, "h", "h", "hora", "horas")
_add_units("tempo", 86400, "dia", "d", "dia", "dias")
_add_units("velocidade", 1 / 3.6, "km/h", "km/h", "kmh")
_add_units("velocidade", 0.44704, "mph", "mph")
_add_units("velocidade", 1, "m/s", "m/s", "mps")


def _convert_units(message: str) -> str | None:
    match = re.search(
        r"(?:converta|converter|transforme)?\s*(-?\d+(?:[.,]\d+)?)\s*([a-z0-9°/]+)\s+(?:para|em|to)\s+([a-z0-9°/]+)\s*\??$",
        _normalize_text(message),
    )
    if not match:
        return None

    amount = float(match.group(1).replace(",", "."))
    source = match.group(2)
    target = match.group(3)
    temperatures = {"c": "c", "°c": "c", "celsius": "c", "f": "f", "°f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}

    if source in temperatures and target in temperatures:
        source_scale = temperatures[source]
        target_scale = temperatures[target]
        celsius = amount if source_scale == "c" else (amount - 32) * 5 / 9 if source_scale == "f" else amount - 273.15
        converted = celsius if target_scale == "c" else celsius * 9 / 5 + 32 if target_scale == "f" else celsius + 273.15
        return f"{_format_number(amount)} °{source_scale.upper()} = {_format_number(converted)} °{target_scale.upper()}."

    source_info = _UNIT_FACTORS.get(source)
    target_info = _UNIT_FACTORS.get(target)
    if source_info is None or target_info is None or source_info[0] != target_info[0]:
        return None

    converted = amount * source_info[1] / target_info[1]
    return f"{_format_number(amount)} {source_info[2]} = {_format_number(converted)} {target_info[2]}."


def _statistics_tool(normalized: str) -> str | None:
    match = re.fullmatch(r"(?:calcule\s+)?(media|mediana|soma|maximo|minimo)\s+(?:de\s+)?(.+)", normalized)
    if not match:
        return None

    kind, raw_values = match.groups()
    try:
        values = [float(item.strip().replace(",", ".")) for item in re.split(r"[;,]", raw_values)]
    except ValueError:
        return None
    if not values or len(values) > 500 or not all(math.isfinite(value) for value in values):
        return "Envie de 1 a 500 números válidos. Separe-os por vírgula ou ponto e vírgula."

    result = {
        "media": statistics.mean,
        "mediana": statistics.median,
        "soma": sum,
        "maximo": max,
        "minimo": min,
    }[kind](values)
    return f"{kind.capitalize()}: {_format_number(float(result))}."


def try_local_tool(message: str) -> str | None:
    """Executa ferramentas locais restritas; retorna None para perguntas ao Ollama."""
    original = message.strip()
    normalized = _normalize_text(original)

    if normalized in {"ajuda", "ajuda ferramentas", "o que voce sabe fazer", "quais ferramentas"}:
        return (
            "Ferramentas locais: contas e porcentagens; média, mediana, soma, máximo e mínimo; "
            "conversão de temperatura, comprimento, massa, volume, dados, tempo e velocidade; "
            "data, horário e dias entre datas; contagem e transformação de texto; IMC e áreas; "
            "número aleatório, dado, moeda e senha aleatória. Você também pode anexar ou tirar fotos "
            "para analisar seu jardim e pedir uma imagem-conceito com “crie uma imagem de...”. Exemplos: "
            "“converta 5 km para milhas”, “média de 4, 8, 10”, “conte palavras: texto”."
        )

    calculation = try_calculate_message(original)
    if calculation is not None:
        return calculation

    if re.search(r"\b(que horas|hora agora|horario atual)\b", normalized):
        return f"Agora são {datetime.now().astimezone().strftime('%H:%M:%S')} (horário local)."
    if re.search(r"\b(que dia e hoje|qual a data|data de hoje|que data e hoje)\b", normalized):
        now = datetime.now().astimezone()
        return f"Hoje é {now.strftime('%d/%m/%Y')} ({now.strftime('%A')}) e são {now.strftime('%H:%M')} no horário local."

    match = re.search(r"(?:dia da semana de|que dia da semana foi)\s+(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})", normalized)
    if match:
        try:
            return f"Essa data caiu em {_parse_date(match.group(1)).strftime('%A')}."
        except ValueError as error:
            return str(error)

    match = re.search(r"(?:daqui a|em)\s+(\d{1,6})\s+dias?", normalized)
    if match and any(word in normalized for word in ("data", "dia", "sera", "será", "daqui")):
        future_date = date.today() + timedelta(days=int(match.group(1)))
        return f"A data será {future_date.strftime('%d/%m/%Y')} ({future_date.strftime('%A')})."

    match = re.search(r"(?:quantos dias ate|dias ate|faltam quantos dias para)\s+(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})", normalized)
    if match:
        try:
            difference = (_parse_date(match.group(1)) - date.today()).days
            return f"Faltam {_format_number(float(difference))} dias." if difference >= 0 else f"Essa data foi há {_format_number(float(abs(difference)))} dias."
        except ValueError as error:
            return str(error)

    match = re.search(r"(?:imc|indice de massa corporal)\s+(?:peso\s*)?([\d.,]+)\s*(?:kg)?\s+(?:e\s*)?(?:altura\s*)?([\d.,]+)\s*m?", normalized)
    if match:
        weight, height = (float(value.replace(",", ".")) for value in match.groups())
        if weight <= 0 or height <= 0 or height > 3:
            return "Informe peso e altura positivos; altura em metros."
        bmi = weight / height**2
        category = "abaixo do peso" if bmi < 18.5 else "peso adequado" if bmi < 25 else "sobrepeso" if bmi < 30 else "obesidade"
        return f"Seu IMC é {_format_number(bmi)} ({category})."

    match = re.search(r"(?:quanto e|calcule)?\s*(\d+(?:[.,]\d+)?)\s*%\s*(?:de|do|da)\s*(\d+(?:[.,]\d+)?)\s*\??$", normalized)
    if match:
        percent, amount = (float(value.replace(",", ".")) for value in match.groups())
        return f"{_format_number(percent)}% de {_format_number(amount)} é {_format_number(percent * amount / 100)}."

    match = re.search(r"(?:desconto de)\s*(\d+(?:[.,]\d+)?)\s*%\s*(?:em|sobre)\s*(\d+(?:[.,]\d+)?)", normalized)
    if match:
        percent, amount = (float(value.replace(",", ".")) for value in match.groups())
        discount = percent * amount / 100
        return f"Desconto: {_format_number(discount)}. Preço final: {_format_number(amount - discount)}."

    match = re.search(r"raiz quadrada de\s*(-?\d+(?:[.,]\d+)?)", normalized)
    if match:
        value = float(match.group(1).replace(",", "."))
        if value < 0:
            return "Não existe raiz quadrada real para um número negativo."
        return f"A raiz quadrada é {_format_number(math.sqrt(value))}."

    match = re.search(r"fatorial de\s*(\d+)", normalized)
    if match:
        value = int(match.group(1))
        if value > 500:
            return "Para manter o cálculo rápido, aceito fatoriais até 500."
        return f"{value}! = {math.factorial(value)}."

    match = re.search(r"converta\s+(\d+)\s+para\s+(binario|octal|hexadecimal|hex)", normalized)
    if match:
        value = int(match.group(1))
        base = {"binario": 2, "octal": 8, "hexadecimal": 16, "hex": 16}[match.group(2)]
        converted = {2: bin, 8: oct, 16: hex}[base](value)
        return f"{value} em base {base}: {converted[2:]}."

    match = re.search(r"(?:area do retangulo|area retangulo)\s+([\d.,]+)\s*(?:por|x)\s*([\d.,]+)", normalized)
    if match:
        first, second = (float(value.replace(",", ".")) for value in match.groups())
        return f"A área do retângulo é {_format_number(first * second)} unidades quadradas."

    match = re.search(r"(?:area do circulo|area circulo)\s+(?:raio\s*)?([\d.,]+)", normalized)
    if match:
        radius = float(match.group(1).replace(",", "."))
        return f"A área do círculo é {_format_number(math.pi * radius**2)} unidades quadradas."

    stats = _statistics_tool(normalized)
    if stats is not None:
        return stats

    converted = _convert_units(original)
    if converted is not None:
        return converted

    match = re.search(r"(?:sorteie|escolha|gere um numero aleatorio entre)\s*(-?\d+)\s+(?:e|a)\s*(-?\d+)", normalized)
    if match:
        start, end = (int(value) for value in match.groups())
        if end < start:
            start, end = end, start
        if end - start > 10**12:
            return "O intervalo é grande demais; use uma diferença de até 1 trilhão."
        return f"Número sorteado: {start + secrets.randbelow(end - start + 1)}."

    if re.search(r"\b(jogue|role|lance)\s+(?:um\s+)?dado\b", normalized):
        return f"O dado caiu em {secrets.randbelow(6) + 1}."
    if normalized in {"cara ou coroa", "jogue uma moeda", "lance uma moeda"}:
        return "Deu cara." if secrets.randbelow(2) == 0 else "Deu coroa."

    match = re.search(r"(?:gere|crie)\s+(?:uma\s+)?senha\s+(?:de|com)\s+(\d+)\s+caracteres?", normalized)
    if match:
        length = int(match.group(1))
        if not 8 <= length <= 128:
            return "Escolha entre 8 e 128 caracteres para uma senha."
        alphabet = string.ascii_letters + string.digits + "!@#$%&*+-_?"
        return "Senha aleatória: " + "".join(secrets.choice(alphabet) for _ in range(length))

    text_prefixes = {
        "conte palavras": "palavras",
        "conte os caracteres": "caracteres",
        "conte caracteres": "caracteres",
        "coloque em maiusculas": "upper",
        "transforme em maiusculas": "upper",
        "coloque em minusculas": "lower",
        "transforme em minusculas": "lower",
        "coloque em titulo": "title",
        "inverta texto": "reverse",
        "remova espacos extras": "spaces",
        "gere slug": "slug",
    }
    prefix, separator, body = original.partition(":")
    action = text_prefixes.get(_normalize_text(prefix.strip())) if separator else None
    if action:
        if action == "palavras":
            return f"O texto tem {len(body.split())} palavras."
        if action == "caracteres":
            return f"O texto tem {len(body)} caracteres ({len(body.replace(' ', ''))} sem espaços)."
        if action == "upper":
            return body.upper()
        if action == "lower":
            return body.lower()
        if action == "title":
            return body.title()
        if action == "reverse":
            return body[::-1]
        if action == "spaces":
            return " ".join(body.split())
        if action == "slug":
            slug = re.sub(r"[^a-z0-9]+", "-", _normalize_text(body)).strip("-")
            return slug or "O texto não contém letras nem números."

    return None


def ask_local_ai(
    message: str,
    history: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    timeout: int = 90,
    images: list[str] | None = None,
) -> str:
    """Conversa com uma instância local do Ollama."""
    system_prompt = (
        "Você é um assistente pessoal que conversa em português do Brasil. "
        "Seja claro e prestativo. Você não controla o computador, não executa "
        "comandos e não deve afirmar que fez ações fora desta conversa."
    )
    if images:
        system_prompt += (
            " Ao analisar fotos de jardins, descreva apenas o que consegue observar. "
            "Se não tiver certeza sobre uma planta ou doença, deixe isso explícito "
            "e faça recomendações cautelosas."
        )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(history[-12:])
    user_message: dict[str, Any] = {"role": "user", "content": message}
    if images:
        user_message["images"] = images
    messages.append(user_message)

    body = json.dumps(
        {
            "model": model.strip() or DEFAULT_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": 0 if images else "5m",
            "options": {
                "num_ctx": 8192,
                "num_predict": 2048,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            },
        }
    ).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        try:
            error_payload = json.loads(details)
            if isinstance(error_payload, dict):
                details = str(error_payload.get("error", details))
        except json.JSONDecodeError:
            pass

        if error.code == 404:
            detail_message = f" Detalhe: {details}" if details else ""
            raise RuntimeError(
                f"O modelo “{model}” não está disponível.{detail_message} "
                "No terminal, execute: "
                f"ollama pull {model}"
            ) from error

        detail_message = f": {details}" if details else "."
        raise RuntimeError(
            f"Ollama respondeu com erro HTTP {error.code}{detail_message}"
        ) from error

    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Não encontrei o Ollama em http://127.0.0.1:11434. "
            "Instale e inicie o Ollama para ativar a conversa com IA."
        ) from error

    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("A resposta do Ollama veio em um formato inválido.") from error

    content = payload.get("message", {}).get("content", "")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("O Ollama não retornou uma resposta.")

    return content.strip()


def append_conversation(path: Path, role: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    label = "Você" if role == "user" else "Boneco"

    with path.open("a", encoding="utf-8") as conversation:
        conversation.write(f"[{timestamp}] {label}: {message}\n\n")