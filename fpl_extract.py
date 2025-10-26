import os
import re

def parse_waypoints(raw_route):
    """
    Recibe una cadena de ruta cruda (raw_route) y devuelve una lista de puntos válidos.
    """
    # Divide el texto en palabras
    tokens = re.split(r'\s+', raw_route.strip())

    waypoints = []
    for token in tokens:
        # Si contiene una barra como LIXAS/N0450F370, extraemos solo lo anterior
        if "/" in token:
            point, suffix = token.split("/", 1)
            if not re.fullmatch(r"[NMFS]\d{3,4}F\d{3}", point):  # asegúrate de que 'point' no es solo nivel
                waypoints.append(point)
        # Si es un nivel de vuelo o velocidad solo (ej: N0450F370), ignoramos
        elif re.fullmatch(r"[NMFS]\d{3,4}F\d{3}", token):
            continue
        else:
            waypoints.append(token)

    return waypoints


# Ruta de archivos
folder_path = r"D:\Maestria\Proyecto_Final\Historical_Data"

# Regexs
fpl_pattern = re.compile(r"\(FPL-[^)]+\)", re.DOTALL)
callsign_pattern = re.compile(r"\(FPL-([A-Z0-9]+)")
departure_info_pattern = re.compile(r"-([A-Z]{4})(\d{4})")
route_pattern = re.compile(r"-[A-Z]\d{3,5}[A-Z]\d{3,5} (.+?)(?=\n-[A-Z]{3,5}\d{3,5})", re.DOTALL)
arrival_info_pattern = re.compile(r"\n-([A-Z]{3,5})(\d{3,5})")


for filename in os.listdir(folder_path):
    if filename.startswith(".HISTORICAL.2023.09.11"):
        file_path = os.path.join(folder_path, filename)
        print(f"\n📄 Procesando archivo: {filename}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

            fpl_matches = fpl_pattern.findall(content)
            print(f"🧾 Planes de vuelo encontrados: {len(fpl_matches)}")

            for fpl_msg in fpl_matches:
                # Callsign
                callsign_match = callsign_pattern.search(fpl_msg)
                callsign = callsign_match.group(1) if callsign_match else "N/A"

                # Departure info
                departure_match = departure_info_pattern.search(fpl_msg)
                departure_airport = departure_match.group(1) if departure_match else "N/A"
                departure_time = departure_match.group(2) if departure_match else "N/A"

                # Raw route
                route_match = route_pattern.search(fpl_msg)
                raw_route = route_match.group(1).replace("\n", " ").strip() if route_match else "N/A"

                #Arrival info
                arrival_match = arrival_info_pattern.search(fpl_msg, route_match.end() if route_match else 0)
                arrival_airport = arrival_match.group(1) if arrival_match else "N/A"
                arrival_time = arrival_match.group(2) if arrival_match else "N/A"

                print(f"✈️ Callsign: {callsign} | 🛫 {departure_airport} ⏰ {departure_time} | 🧭 Ruta: {raw_route}")
                print(f"🛬 Destino: {arrival_airport} ⏰ {arrival_time}")

                print(f"✈️ Callsign: {callsign}")
                if raw_route == "N/A":
                    print("  ⚠️ Ruta no encontrada o malformada.")
                    continue

                #waypoints = parse_waypoints(raw_route)
                #for i, wp in enumerate(waypoints, 1):
                #    print(f"  {i}. {wp}")
                
