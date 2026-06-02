from dataclasses import dataclass, field


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
    familia: str = ""
    orden: str = ""
    descripcion: str = ""
    distribucion: str = ""
    img: str = ""
    habitat: list[str] = field(default_factory=list)
    dieta: list[str] = field(default_factory=list)
    estado_conservacion: str = ""
    longitud_cm: float | int | None = None
    peso_g_min: float | int | None = None
    peso_g_max: float | int | None = None
    
    def __str__(self):
        habitat = ", ".join(self.habitat or []) if self.habitat else ""
        dieta = ", ".join(self.dieta or []) if self.dieta else ""
        peso = ""
        if self.peso_g_min is not None or self.peso_g_max is not None:
            peso = f"{self.peso_g_min if self.peso_g_min is not None else '?'} - {self.peso_g_max if self.peso_g_max is not None else '?'} g"

        return (
            f"Nombre común (inglés): {self.nombre_comun_ingles}\n"
            f"Nombre común (español): {self.nombre_comun_espanol}\n"
            f"Nombre científico: {self.nombre_cientifico}\n"
            f"Familia: {self.familia}\n"
            f"Orden: {self.orden}\n"
            f"Descripción: {self.descripcion}\n"
            f"Distribución: {self.distribucion}\n"
            f"Hábitat: {habitat}\n"
            f"Dieta: {dieta}\n"
            f"Estado de conservación: {self.estado_conservacion}\n"
            f"Longitud: {self.longitud_cm} cm\n"
            f"Peso: {peso}\n"
            f"Vocalizaciones: {self.vocalizaciones.descripcion}\n"
            f"Tipo de vocalización: {', '.join(self.vocalizaciones.tipo_vocalizacion)}\n"
            f"Frecuencias (Hz): {self.vocalizaciones.frecuencias_hz.rango_principal.min} - "
            f"{self.vocalizaciones.frecuencias_hz.rango_principal.max}\n"
            f"Frecuencia dominante: {self.vocalizaciones.frecuencias_hz.frecuencia_dominante}\n"
            f"Notas: {self.vocalizaciones.frecuencias_hz.notas}"
        )