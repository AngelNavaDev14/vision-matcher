import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class MatchResult:
    found: bool
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    confidence: float = 0.0
    label: str = ""

def escalas_por_resolucion(h_screen: int) -> list[float]:
    base = 1080
    exact = h_screen / base
    # Prueba la escala exacta + margen de +-15% en pasos finos
    scales = sorted(set(
        [exact] +
        list(np.linspace(exact * 0.85, exact * 1.15, 12))
    ))
    return scales

class TemplateMatcher:
    def __init__(self, templates_dir: str = "templates", threshold: float = 0.75):
        self.threshold = threshold
        self.templates_dir = Path(templates_dir)
        self.templates: dict[str, list[np.ndarray]] = {}
        self._load_templates()

    def _load_templates(self):
        if not self.templates_dir.exists():
            self.templates_dir.mkdir()
            return
        for img_path in self.templates_dir.glob("*.png"):
            label = img_path.stem.split("_")[0].lower()
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if label not in self.templates:
                    self.templates[label] = []
                self.templates[label].append(img)

    def match(self, screenshot: np.ndarray, label: str) -> Optional[MatchResult]:
        if label.lower() not in self.templates:
            return None

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        h_screen, w_screen = gray.shape
        best = MatchResult(found=False, label=label)

        scales = escalas_por_resolucion(h_screen)

        for template in self.templates[label.lower()]:
            th, tw = template.shape

            for scale in scales:
                new_w = int(tw * scale)
                new_h = int(th * scale)
                if new_w > w_screen or new_h > h_screen:
                    continue
                if new_w < 10 or new_h < 10:
                    continue

                resized = cv2.resize(template, (new_w, new_h))
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best.confidence:
                    best = MatchResult(
                        found=max_val >= self.threshold,
                        x=int(max_loc[0]),
                        y=int(max_loc[1]),
                        w=new_w,
                        h=new_h,
                        confidence=round(float(max_val), 4),
                        label=label
                    )

                # Si ya tenemos confianza perfecta, no seguimos
                if best.confidence >= 0.99:
                    return best

        return best if best.found else None

    def match_any(self, screenshot: np.ndarray) -> list[MatchResult]:
        results = []
        for label in self.templates:
            r = self.match(screenshot, label)
            if r:
                results.append(r)
        return results
