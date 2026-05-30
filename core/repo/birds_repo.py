from pathlib import Path
from typing import Literal

from core.dto.cards_bird import (
    BirdInfo
)
from config.routes_path import RoutesPath
from core.dto.cards_bird import BirdInfo
from core.birds_data_loader import BirdDataLoader
class BirdRepository:

    def __init__(
        self,
        env: Literal['training', 'prediction'] = 'training',
    ):
        
        self._birds = BirdDataLoader(RoutesPath.AVES_INFO).load()
        self._env = env
        path = Path(RoutesPath.BANK_AUDIOS_NORMALIZED) if env == 'training' else Path(RoutesPath.PROCESSED_AUDIOS)
        self.repository_path = path

    def get_by_species(
        self,
        english_name: str
    ) -> BirdInfo | None:

        ##filter by scientific name
        for bird in self._birds:
            if bird.nombre_comun_ingles == english_name:
                return bird
        
    def get_specie_by_audio_path(
        self,
        audio_path: str
    ) -> BirdInfo | None:

        path = Path(audio_path)
        specie_name = path.parent.name
        return self.get_by_species(specie_name)
    
    def get_audios_by_species(
        self,
    ) -> dict[str, list[Path]]:

        audios_by_species: dict[str, list[Path]] = {}

        for bird in self._birds:
            specie_name = bird.nombre_comun_ingles
            specie_audios = list(self.repository_path.glob(f"{specie_name}/*.wav"))
            audios_by_species[specie_name] = specie_audios

        return audios_by_species