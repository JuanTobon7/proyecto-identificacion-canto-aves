import json
from pathlib import Path

from core.dto.cards_bird import (
    BirdInfo,
    Vocalizations,
    VocalFrequencies,
    FrequencyRange,
)


class BirdDataLoader:

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)

    def load(self) -> list[BirdInfo]:

        with open(
            self.json_path,
            "r",
            encoding="utf-8"
        ) as fh:

            raw = json.load(fh)

        birds: list[BirdInfo] = []

        for bird in raw["aves"]:

            rango = FrequencyRange(
                min=bird["vocalizaciones"]
                ["frecuencias_Hz"]
                ["rango_principal"]["min"],

                max=bird["vocalizaciones"]
                ["frecuencias_Hz"]
                ["rango_principal"]["max"],
            )

            vocal_freq = VocalFrequencies(
                rango_principal=rango,
                frecuencia_dominante=bird[
                    "vocalizaciones"
                ]["frecuencias_Hz"][
                    "frecuencia_dominante"
                ],
                notas=bird[
                    "vocalizaciones"
                ]["frecuencias_Hz"][
                    "notas"
                ],
            )

            vocal = Vocalizations(
                descripcion=bird[
                    "vocalizaciones"
                ]["descripcion"],

                tipo_vocalizacion=bird[
                    "vocalizaciones"
                ][
                    "tipo_vocalización"
                ],

                frecuencias_hz=vocal_freq
            )

            birds.append(
                BirdInfo(
                    nombre_comun_ingles=bird[
                        "nombre_comun_ingles"
                    ],
                    nombre_comun_espanol=bird[
                        "nombre_comun_espanol"
                    ],
                    nombre_cientifico=bird[
                        "nombre_cientifico"
                    ],
                    vocalizaciones=vocal
                )
            )

        return birds