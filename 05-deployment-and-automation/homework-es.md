## Homework 5: Automatización (cohorte 2026)

> Entrega: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05 (vence el **09/11/2026, 22:59 UTC**)
> Enunciado original: todavía no publicado (al 21/10/2026 no está en `cohorts/2026/` ni tiene preguntas en la plataforma).
> Código: esta carpeta (pipeline del Módulo 5 funcionando localmente)

En 2024 y 2025 el Módulo 5 no tuvo homework; en 2026 la plataforma ya creó "Homework 5: Automation".

## Lo que ya está listo

- El sistema de automatización del curso (`main.py` + `scripts/`) corre de punta a punta en Windows con `uv`:
  descarga de 186 acciones + índices + FRED → ~390 features con TA-Lib → Random Forest → predicciones del último día.
- Dos regímenes: completo (`main.py`) y rápido desde disco (`main.py --fast`).
- Ejemplos de cron (Linux) y del Programador de tareas (Windows) en `README.md`.

## Pendiente

- [ ] Traducir el enunciado cuando se publique y resolver cada pregunta sobre este pipeline.
- [ ] Post de LinkedIn del módulo.
