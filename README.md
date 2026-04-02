# Vision Matcher — Template Matching con OpenCV

Reemplaza OCR costoso con template matching eficiente para detectar elementos de UI en capturas de pantalla.

## Resultados

- Detección de "Kickoff" y "Final" en screenshots de Madden 2026
- Soporte multi-resolución (1080p, 1440p, 4K, ultrawide)
- Confianza de 1.0 en match exacto, 0 falsos positivos

## API

POST /detect       — detecta un label específico
POST /detect_all   — detecta todos los templates cargados
GET  /health       — estado del servicio

## Stack

Python + OpenCV 4.13 + FastAPI
