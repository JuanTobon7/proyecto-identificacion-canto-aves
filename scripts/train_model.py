from config.routes_path import RoutesPath

import argparse

def main():
    parser = argparse.ArgumentParser(description="Análisis y limpieza de corpus de audio.")
    
    parser.add_argument(
        "model --fir"
        default= "butterworth",   
    )
    
    