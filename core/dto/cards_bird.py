from dataclasses import dataclass


@dataclass
class FrequencyRange:
    min: float
    max: float


@dataclass
class VocalFrequencies:
    rango_principal: FrequencyRange
    frecuencia_dominante: float
    notas: str


@dataclass
class Vocalizations:
    descripcion: str
    tipo_vocalizacion: list[str]
    frecuencias_hz: VocalFrequencies


@dataclass
class BirdInfo:
    nombre_comun_ingles: str
    nombre_comun_espanol: str
    nombre_cientifico: str
    vocalizaciones: Vocalizations
    
    def __str__(self):
        return (f"Nombre común (inglés): {self.nombre_comun_ingles}\n"
                f"Nombre común (español): {self.nombre_comun_espanol}\n"
                f"Nombre científico: {self.nombre_cientifico}\n"
                f"Vocalizaciones: {self.vocalizaciones.descripcion}\n"
                f"Tipo de vocalización: {', '.join(self.vocalizaciones.tipo_vocalizacion)}\n"
                f"Frecuencias (Hz): {self.vocalizaciones.frecuencias_hz.rango_principal.min} - "
                f"{self.vocalizaciones.frecuencias_hz.rango_principal.max}\n"
                f"Frecuencia dominante: {self.vocalizaciones.frecuencias_hz.frecuencia_dominante}\n"
                f"Notas: {self.vocalizaciones.frecuencias_hz.notas}")