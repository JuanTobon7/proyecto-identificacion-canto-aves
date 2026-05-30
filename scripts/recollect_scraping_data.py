import re
import time
import random
import requests
import pandas as pd
import json

from pathlib import Path
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, parse_qs

# =====================================================
# CONFIG
# =====================================================

BASE_URL = (
    "https://media.ebird.org/catalog?"
    "birdOnly=true&regionCode=CO"
    "&mediaType=audio&view=grid"
)

OUTPUT_DIR = "dataset_aves"

HEADLESS = False

MAX_LOAD_MORE = 30

MAX_AUDIOS_PER_SPECIES = 200

MIN_WAIT = 1.5
MAX_WAIT = 4.0

DOWNLOAD_TIMEOUT = 60

# Track downloaded audios to avoid duplicates
DOWNLOADED_ML_IDS = set()
DOWNLOADED_ML_IDS_FILE = "downloaded_ml_ids.json"

# CSS Selectors
RESULTS_GRID_CARD_SELECTOR = "li.ResultsGrid-card"

# =====================================================
# HELPERS
# =====================================================

def load_downloaded_ids():
    """Load previously downloaded ML IDs"""
    global DOWNLOADED_ML_IDS
    if Path(DOWNLOADED_ML_IDS_FILE).exists():
        try:
            with open(DOWNLOADED_ML_IDS_FILE, 'r') as f:
                DOWNLOADED_ML_IDS = set(json.load(f))
                print(f"[INFO] Cargados {len(DOWNLOADED_ML_IDS)} IDs previos")
        except Exception:
            DOWNLOADED_ML_IDS = set()
    else:
        DOWNLOADED_ML_IDS = set()


def save_downloaded_ids():
    """Save downloaded ML IDs to file"""
    with open(DOWNLOADED_ML_IDS_FILE, 'w') as f:
        json.dump(list(DOWNLOADED_ML_IDS), f, indent=2)


def clean_name(name):
    name = re.sub(r'[<>:"/\\|?*]', '', str(name))
    name = re.sub(r"\s+", "_", name.strip())
    return name


def build_session():

    session = requests.Session()

    session.headers.update({
        "User-Agent":
        "AcademicBirdDataset/1.0"
    })

    return session


def safe_text(locator):
    try:
        return locator.inner_text().strip()
    except Exception:
        return ""


def get_taxon_code_from_species_page(page, species_name):
    """
    Navega a la página de la especie y extrae el taxonCode 
    desde el enlace de audios en la tabla de estadísticas
    """
    try:
        # Buscar el link a la página de la especie
        species_link = page.locator(
            f"a:has-text('{species_name}')"
        ).first
        
        if not species_link.is_visible():
            print(f"[WARNING] No se encontró link para {species_name}")
            return None
        
        # Obtener la URL del link
        species_url = species_link.get_attribute("href")
        if not species_url:
            print(f"[WARNING] No se encontró href para {species_name}")
            return None
        
        # Hacer click en el link
        species_link.click()
        time.sleep(random.uniform(3, 5))
        
        # Esperar a que cargue la página
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Buscar la tabla de estadísticas y el enlace de audios
        audio_link = page.locator(
            "a[href*='mediaType=audio']"
        ).first
        
        if not audio_link.is_visible():
            print(f"[WARNING] No se encontró enlace de audios para {species_name}")
            return None
        
        audio_url = audio_link.get_attribute("href")
        print(f"[INFO] Audio URL encontrada: {audio_url}")
        
        # Extraer el taxonCode de la URL
        if "taxonCode=" in audio_url:
            taxon_code = audio_url.split("taxonCode=")[1].split("&")[0]
            print(f"[OK] TaxonCode extraído: {taxon_code}")
            return taxon_code
        
        return None
        
    except Exception as e:
        print(f"[ERROR] Error extrayendo taxonCode: {e}")
        return None


# =====================================================
# DOWNLOAD AUDIO
# =====================================================

def download_audio(
    session,
    asset_id,
    species_folder
):

    if asset_id in DOWNLOADED_ML_IDS:
        print(
            f"[SKIP] ML{asset_id} (ya descargado)"
        )
        return False

    output_file = (
        species_folder /
        f"ML{asset_id}.mp3"
    )

    if output_file.exists():
        print(
            f"[SKIP] ML{asset_id} (archivo existe)"
        )
        DOWNLOADED_ML_IDS.add(asset_id)
        return False

    audio_url = (
        f"https://cdn.download.ams.birds.cornell.edu/"
        f"api/v1/asset/{asset_id}"
    )

    try:

        response = session.get(
            audio_url,
            timeout=DOWNLOAD_TIMEOUT,
            stream=True
        )

        if response.status_code != 200:
            print(
                f"[ERROR {response.status_code}] "
                f"ML{asset_id}"
            )
            return False

        with open(
            output_file,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=8192
            ):
                if chunk:
                    f.write(chunk)

        print(
            f"[OK] {output_file}"
        )
        DOWNLOADED_ML_IDS.add(asset_id)
        return True

    except Exception as e:
        print(
            f"[ERROR] "
            f"ML{asset_id}: {e}"
        )
        return False


def download_species_audios(
    page,
    session,
    common_name,
    taxon_code,
    metadata
):
    """
    Descarga todos los audios de una especie específica
    navegando a su página de catálogo
    """
    
    species_catalog_url = (
        f"https://media.ebird.org/catalog?"
        f"birdOnly=true&taxonCode={taxon_code}"
        f"&mediaType=audio&view=grid"
    )
    
    print(f"\n[SPECIES] Descargando audios de {common_name}")
    print(f"[URL] {species_catalog_url}")
    
    try:
        page.goto(
            species_catalog_url,
            wait_until="networkidle"
        )
        
        time.sleep(random.uniform(2, 3))
        
        # Cargar más audios
        for i in range(MAX_LOAD_MORE):
            try:
                button = page.locator(
                    "button:has-text('More results')"
                )
                
                if not button.is_visible():
                    print(
                        f"[INFO] Sin más resultados para {common_name}"
                    )
                    break
                
                button.click()
                wait = random.uniform(1.5, 3)
                print(
                    f"Load more {i+1} ({wait:.1f}s)"
                )
                time.sleep(wait)
                
            except Exception:
                break
        
        # Extraer todas las tarjetas de audios
        cards = page.locator(
            RESULTS_GRID_CARD_SELECTOR
        )
        
        total = cards.count()
        print(
            f"[INFO] Total audios encontrados: {total}"
        )
        
        species_folder = (
            Path(OUTPUT_DIR) /
            clean_name(common_name)
        )
        
        species_folder.mkdir(
            parents=True,
            exist_ok=True
        )
        
        downloaded_count = 0
        
        for i in range(total):
            try:
                # Detener si ya hemos descargado el máximo
                if downloaded_count >= MAX_AUDIOS_PER_SPECIES:
                    print(
                        f"[INFO] Límite alcanzado: {MAX_AUDIOS_PER_SPECIES} audios por especie"
                    )
                    break
                
                card = cards.nth(i)
                
                asset_id = card.locator(
                    "div[data-asset-id]"
                ).get_attribute(
                    "data-asset-id"
                )
                
                date = safe_text(
                    card.locator(
                        "time"
                    )
                )
                
                location = safe_text(
                    card.locator(
                        ".userDateLoc span"
                    )
                )
                
                print(
                    f"\n[{i+1}/{total}] ML{asset_id}"
                )
                
                if download_audio(
                    session,
                    asset_id,
                    species_folder
                ):
                    downloaded_count += 1
                    
                    metadata.append({
                        "asset_id": asset_id,
                        "common_name": common_name,
                        "scientific_name": "",
                        "date": date,
                        "location": location,
                        "taxon_code": taxon_code,
                    })
                
                wait = random.uniform(
                    MIN_WAIT,
                    MAX_WAIT
                )
                time.sleep(wait)
                
            except Exception as e:
                print(
                    f"[CARD ERROR] {e}"
                )
        
        print(
            f"[SUMMARY] {downloaded_count} audios descargados de {common_name}"
        )
        return downloaded_count
        
    except Exception as e:
        print(
            f"[ERROR] Error descargando audios de {common_name}: {e}"
        )
        return 0


# =====================================================
# SCRAPER
# =====================================================

def scrape():

    Path(
        OUTPUT_DIR
    ).mkdir(exist_ok=True)
    
    # Cargar IDs previamente descargados
    load_downloaded_ids()

    metadata = []

    session = build_session()

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page()

        print("Abriendo eBird...")

        page.goto(
            BASE_URL,
            wait_until="networkidle"
        )

        print(
            "Cargando más resultados..."
        )

        for i in range(
            MAX_LOAD_MORE
        ):

            try:

                button = page.locator(
                    "button:has-text('More results')"
                )

                if not button.is_visible():
                    print(
                        "No hay más resultados."
                    )
                    break

                button.click()

                wait = random.uniform(
                    2,
                    4
                )

                print(
                    f"Load more {i+1} "
                    f"(esperando {wait:.1f}s)"
                )

                time.sleep(wait)

            except Exception:
                break

        print(
            "Extrayendo tarjetas..."
        )

        cards = page.locator(
            RESULTS_GRID_CARD_SELECTOR
        )

        total = cards.count()

        print(
            f"Total cards: {total}"
        )

        for i in range(total):

            try:

                card = cards.nth(i)

                asset_id = card.locator(
                    "div[data-asset-id]"
                ).get_attribute(
                    "data-asset-id"
                )

                common_name = safe_text(
                    card.locator(
                        ".Species-common"
                    )
                )

                print(
                    f"\n[{i+1}/{total}] "
                    f"{common_name}"
                    f" - ML{asset_id}"
                )

                # Buscar si ya hemos procesado esta especie
                if any(m.get("common_name") == common_name for m in metadata):
                    print(
                        f"[SKIP] {common_name} ya procesada"
                    )
                    wait = random.uniform(
                        MIN_WAIT,
                        MAX_WAIT
                    )
                    time.sleep(wait)
                    continue

                # Extraer el taxonCode de la página de estadísticas
                taxon_code = get_taxon_code_from_species_page(
                    page,
                    common_name
                )
                
                if taxon_code:
                    # Descargar todos los audios de esta especie
                    download_species_audios(
                        page,
                        session,
                        common_name,
                        taxon_code,
                        metadata
                    )
                    
                    # Volver a la página principal
                    page.goto(
                        BASE_URL,
                        wait_until="networkidle"
                    )
                    time.sleep(random.uniform(2, 3))
                    
                    # Recargar las tarjetas después de volver
                    cards = page.locator(
                        RESULTS_GRID_CARD_SELECTOR
                    )
                else:
                    print(
                        f"[WARNING] No se pudo extraer taxonCode para {common_name}"
                    )

                wait = random.uniform(
                    MIN_WAIT,
                    MAX_WAIT
                )

                time.sleep(wait)

            except Exception as e:

                print(
                    f"[CARD ERROR] "
                    f"{e}"
                )

        browser.close()

    # Guardar metadata
    if metadata:
        pd.DataFrame(
            metadata
        ).to_csv(
            "metadata.csv",
            index=False
        )

    # Guardar IDs descargados
    save_downloaded_ids()

    print("\nFINALIZADO")


if __name__ == "__main__":
    scrape()