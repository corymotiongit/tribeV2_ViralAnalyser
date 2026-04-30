from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisModeProfile:
    key: str
    label: str
    short_label: str
    description: str
    ui_note: str
    min_segment_word_probability: float
    max_segment_no_speech_probability: float
    min_total_words: int
    min_total_speech_seconds: float
    min_average_word_probability: float
    drop_activation_z_threshold: float
    drop_novelty_z_threshold: float
    max_drop_markers: int
    recommendation_cutoff: int
    comparison_spread_note: str
    max_action_items: int


ANALYSIS_MODE_PROFILES: dict[str, AnalysisModeProfile] = {
    "deep": AnalysisModeProfile(
        key="deep",
        label="Глубокий анализ",
        short_label="Deep",
        description="Разжёвывает максимум информации: подробнее объясняет график, слабые места и разницу между версиями.",
        ui_note="Подходит, когда нужно внимательно понять, за счёт чего ролик выигрывает или проседает, а не просто получить короткий список правок.",
        min_segment_word_probability=0.28,
        max_segment_no_speech_probability=0.58,
        min_total_words=2,
        min_total_speech_seconds=0.4,
        min_average_word_probability=0.62,
        drop_activation_z_threshold=-0.65,
        drop_novelty_z_threshold=-0.2,
        max_drop_markers=6,
        recommendation_cutoff=66,
        comparison_spread_note="Режим более чувствительный: он показывает не только грубого победителя, но и за счёт каких показателей одна версия обходит другую.",
        max_action_items=6,
    ),
    "simplified": AnalysisModeProfile(
        key="simplified",
        label="Упрощённый",
        short_label="Simple",
        description="Пишет простым языком: что оставить, что поправить и на какой секунде именно это делать.",
        ui_note="Подходит, когда нужен короткий рабочий вывод без лишней аналитики: открыл отчёт, понял, что менять, и сразу пошёл править.",
        min_segment_word_probability=0.36,
        max_segment_no_speech_probability=0.42,
        min_total_words=3,
        min_total_speech_seconds=0.6,
        min_average_word_probability=0.72,
        drop_activation_z_threshold=-0.85,
        drop_novelty_z_threshold=-0.42,
        max_drop_markers=4,
        recommendation_cutoff=60,
        comparison_spread_note="Режим более прикладной: он выводит не всё подряд, а только различия, которые проще всего превратить в следующую правку.",
        max_action_items=4,
    ),
}

DEFAULT_ANALYSIS_MODE = "deep"


def get_analysis_mode_profile(key: str | None) -> AnalysisModeProfile:
    if not key:
        return ANALYSIS_MODE_PROFILES[DEFAULT_ANALYSIS_MODE]
    return ANALYSIS_MODE_PROFILES.get(key, ANALYSIS_MODE_PROFILES[DEFAULT_ANALYSIS_MODE])
