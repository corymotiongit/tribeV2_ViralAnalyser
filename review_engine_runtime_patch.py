from __future__ import annotations

from typing import Any

import review_engine


def _focus_valid_indices(timestamps: list[float]) -> list[int]:
    if len(timestamps) <= 4:
        return list(range(len(timestamps)))

    start = float(timestamps[0])
    end = float(timestamps[-1])
    duration = max(0.0, end - start)
    edge_buffer = max(3.0, min(4.0, duration * 0.04))
    if duration <= 8.0:
        edge_buffer = min(0.8, max(0.35, duration * 0.05))

    tail_buffer = 5.0 if duration > 8.0 else max(1.0, duration * 0.30)
    upper_bound = end - tail_buffer
    candidates = [
        index
        for index, ts in enumerate(timestamps)
        if (start + edge_buffer) < float(ts) < upper_bound
    ]
    if len(candidates) >= 3:
        return candidates

    middle = [
        index
        for index, ts in enumerate(timestamps[1:-1], start=1)
        if float(ts) <= upper_bound
    ]
    return middle or list(range(len(timestamps)))


ACTION_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "early_response": [
        ("Усиль первый кадр", "Поставь перед этой точкой кадр, где сразу видно главный объект, результат или конфликт."),
        ("Начни с результата", "Покажи итог или самый понятный эффект раньше, а объяснение оставь после него."),
        ("Убери долгий заход", "Если перед этой точкой есть вступление без нового смысла, сократи его до первого действия."),
        ("Подними главный объект", "Сделай объект крупнее или ближе к центру уже в начале слабого окна."),
        ("Дай обещание раньше", "Если ролик продает результат, покажи его пользу до просадки, а не после нее."),
        ("Начни с действия", "Замени спокойный вход на кадр с движением, жестом или заметным изменением."),
        ("Переставь сильный кадр", "Возьми ближайший сильный кадр после просадки и протестируй его раньше."),
        ("Сократи пустой старт", "Убери кадры, где зритель еще не понимает, на что смотреть."),
        ("Покажи контекст сразу", "Добавь короткую визуальную подсказку, чтобы смысл считывался до слабого места."),
        ("Сделай вход резче", "Ускорь первые секунды: меньше паузы, крупнее объект, яснее действие."),
    ],
    "sustain": [
        ("Подрежь затянутый отрезок", "Убери 1-2 секунды перед этой точкой или быстрее переведи ролик к следующему действию."),
        ("Добавь новый поворот", "Перед этой точкой вставь новую деталь, движение или смену плана, чтобы ролик не провисал."),
        ("Собери темп плотнее", "Сожми паузу и оставь только кадры, которые двигают сцену вперед."),
        ("Обнови середину", "Добавь в этот участок новую информацию: реакцию, деталь, результат или изменение действия."),
        ("Сократи повтор", "Если кадр повторяет уже понятную мысль, оставь только самый сильный кусок."),
        ("Смени крупность", "Перед просадкой перейди на другой масштаб: крупный план, общий план или деталь."),
        ("Дай маленький payoff", "Вставь быстрый мини-результат до того, как график начинает падать."),
        ("Переставь событие ближе", "Если важное действие происходит позже, протестируй его на 1-2 секунды раньше."),
        ("Убери нейтральный кадр", "Кадр без новой информации лучше заменить движением или реакцией."),
        ("Разбей длинный план", "Раздели статичный участок короткой сменой ракурса или вставкой детали."),
    ],
    "transition": [
        ("Смени кадр раньше", "Смени план, ракурс или действие раньше, чтобы этот участок не тянулся."),
        ("Добавь визуальный акцент", "Перед этой точкой добавь движение, жест, приближение или смену крупности."),
        ("Убери зависший план", "Если кадр стоит без нового действия, сократи его до первого понятного движения."),
        ("Вставь деталь", "Добавь короткий крупный план детали, чтобы зритель получил новый повод смотреть."),
        ("Поменяй ракурс", "Оставь то же действие, но покажи его с другого угла до начала просадки."),
        ("Добавь реакцию", "Если есть человек, животное или объект в действии, вставь реакцию или последствие."),
        ("Ускорь монтаж", "Проверь более короткую длительность этого плана без изменения смысла сцены."),
        ("Сделай переход заметнее", "Используй движение в кадре или совпадение действия, чтобы смена не выглядела случайной."),
        ("Раздели однообразный кусок", "Внутри длинного фрагмента добавь вторую визуальную фазу: было - стало, до - после."),
        ("Дай текстовую опору", "Если картинка похожа сама на себя, добавь короткую подпись с новым смыслом."),
    ],
    "stability": [
        ("Убери лишнее из кадра", "Оставь один главный объект и убери лишние детали или текст рядом с ним."),
        ("Сделай фокус понятнее", "Подсвети главный объект крупностью, положением в кадре или более чистым фоном."),
        ("Разгрузи композицию", "Убери конкурирующие элементы, чтобы взгляд не распадался между несколькими деталями."),
        ("Спрячь лишний текст", "Если рядом с главным объектом много слов, оставь одну короткую подпись или убери ее совсем."),
        ("Укрупни главный объект", "Сделай важный объект больше, чтобы он не конкурировал с фоном."),
        ("Очисти фон", "Проверь кадр без лишних предметов, бликов или деталей за главным действием."),
        ("Сделай движение понятнее", "Если действие мелкое, покажи его крупнее или повтори в более читаемом ракурсе."),
        ("Убери второй центр внимания", "Оставь один главный фокус, а второстепенный объект затемни, обрежь или вынеси позже."),
        ("Стабилизируй кадр", "Если просадка рядом с тряской или резким сдвигом, протестируй более спокойный фрагмент."),
        ("Отдели объект от фона", "Усиль разницу светом, цветом или рамкой, чтобы главное не сливалось."),
    ],
    "density": [
        ("Подними средний уровень", "Усиль не один пик, а обычные кадры вокруг этой точки: крупнее объект, чище фон, заметнее действие."),
        ("Покажи товар крупнее", "Сделай объект крупнее, усили движение в кадре или добавь контраст."),
        ("Усиль визуальный удар", "Перед этой точкой добавь более яркий кадр, крупный план или заметное действие."),
        ("Сделай кадр контрастнее", "Отдели главный объект от фона светом, цветом или более чистой композицией."),
        ("Добавь движение", "Если кадр статичный, протестируй движение руки, камеры, объекта или смену положения."),
        ("Покажи деталь ближе", "Вставь крупный план детали, на которую зритель должен обратить внимание."),
        ("Убери серый кадр", "Замени нейтральный фрагмент на кадр с более явным действием или эмоцией."),
        ("Сделай пользу видимой", "Если продукт или результат плохо читается, покажи его эффект прямо в кадре."),
        ("Дай визуальный контраст", "Проверь светлый объект на темном фоне, цветовой акцент или более чистую композицию."),
        ("Собери сильнее сцену", "Убери слабые промежуточные кадры и оставь те, где объект, действие и смысл видны сразу."),
    ],
    "speech_start": [
        ("Скажи главное раньше", "Если смысл держится на словах, подай главную фразу до этой точки и сократи немой заход."),
        ("Перенеси фразу вперед", "Поставь ключевую реплику ближе к началу слабого участка."),
        ("Начни с короткой фразы", "Добавь одну понятную реплику до просадки, без длинного объяснения."),
        ("Убери немой заход", "Если первые секунды без слов не работают, сократи их или положи поверх ключевую мысль."),
        ("Синхронизируй слово и кадр", "Пусть важная фраза звучит в тот момент, когда главный объект уже виден."),
    ],
    "pause": [
        ("Убери паузу", "Подрежь пустой промежуток или скажи фразу плотнее, чтобы участок не проседал."),
        ("Сожми речь", "Сократи паузу между словами и оставь только нужную фразу."),
        ("Подтяни подачу", "Сделай фразу короче и ближе к действию в кадре."),
        ("Закрой пустое место", "Если паузу нельзя убрать, перекрой ее действием, реакцией или крупным планом."),
        ("Разрежь длинную фразу", "Раздели речь на короткие куски и поставь каждый рядом с нужным кадром."),
    ],
}


def _action_copy_for_metric(metric_key: str, variant_index: int) -> tuple[str, str]:
    variants = ACTION_VARIANTS.get(metric_key) or ACTION_VARIANTS["sustain"]
    return variants[variant_index % len(variants)]


def _action_metric_candidates(metrics: list[Any], speech_layer: dict[str, Any]) -> list[str]:
    keys = [str(item.key) for item in sorted(metrics, key=lambda metric: metric.score) if getattr(item, "key", "")]
    if speech_layer.get("available") and isinstance(speech_layer.get("speech_start_seconds"), float) and speech_layer["speech_start_seconds"] > 2.0:
        keys.append("speech_start")
    if speech_layer.get("available") and isinstance(speech_layer.get("pause_ratio"), float) and speech_layer["pause_ratio"] > 0.28:
        keys.append("pause")

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered or ["sustain"]


def _build_action_items(
    recommendations: list[str],
    focus_windows: list[Any],
    drop_moments: list[dict[str, Any]],
    speech_layer: dict[str, Any],
    metrics: list[Any],
    profile: Any,
) -> list[dict[str, str]]:
    del recommendations

    targets: list[dict[str, str]] = []
    for item in drop_moments:
        timestamp = str(item.get("timestamp") or "").strip()
        if not timestamp:
            continue
        targets.append(
            {
                "timestamp": timestamp,
                "why": str(item.get("reason") or "").strip(),
            }
        )

    if not targets and focus_windows:
        weak_window = focus_windows[1] if getattr(profile, "key", "") == "simplified" and len(focus_windows) > 1 else focus_windows[0]
        if getattr(profile, "key", "") != "simplified":
            weak_window = next(
                (
                    item
                    for item in focus_windows
                    if "лаб" in str(getattr(item, "label", "")).lower()
                ),
                focus_windows[0],
            )
        targets.append(
            {
                "timestamp": str(getattr(weak_window, "timestamp", "")).strip(),
                "why": str(getattr(weak_window, "summary", "")).strip(),
            }
        )

    metric_keys = _action_metric_candidates(metrics, speech_layer)
    actions: list[dict[str, str]] = []
    used_titles: set[str] = set()
    metric_use_count: dict[str, int] = {}
    for index, target in enumerate(targets[: getattr(profile, "max_action_items", 4)]):
        metric_key = metric_keys[min(index, len(metric_keys) - 1)]
        variant_index = metric_use_count.get(metric_key, 0)
        title, instruction = _action_copy_for_metric(metric_key, variant_index)
        metric_use_count[metric_key] = variant_index + 1
        if title in used_titles:
            for offset in range(1, 5):
                title, instruction = _action_copy_for_metric(metric_key, variant_index + offset)
                if title not in used_titles:
                    break
        used_titles.add(title)
        actions.append(
            {
                "timestamp": target["timestamp"],
                "title": title,
                "instruction": instruction,
                "why": target["why"],
            }
        )

    seen_timestamps: set[str] = set()
    deduped: list[dict[str, str]] = []
    for action in actions:
        timestamp = action["timestamp"]
        if not timestamp or timestamp in seen_timestamps:
            continue
        seen_timestamps.add(timestamp)
        deduped.append(action)
    return deduped[: getattr(profile, "max_action_items", 4)]


def apply() -> None:
    review_engine._focus_valid_indices = _focus_valid_indices
    review_engine._build_action_items = _build_action_items
