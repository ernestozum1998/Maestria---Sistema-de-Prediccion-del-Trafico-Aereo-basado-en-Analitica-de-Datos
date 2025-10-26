-- Usar la base de datos
USE atc_flight_data;

-- Tabla de aeropuertos (sin duplicados)
CREATE TABLE IF NOT EXISTS airports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    icao_code VARCHAR(4) UNIQUE NOT NULL,
    name VARCHAR(100)
);

-- Tabla de aeropuertos (sin duplicados)
CREATE TABLE IF NOT EXISTS airlines (
    icao_prefix VARCHAR(10) UNIQUE NOT NULL PRIMARY KEY,
    airline_name VARCHAR(100),
    flight_count int 
);

-- Tabla para metadatos de archivos procesados
CREATE TABLE IF NOT EXISTS metadata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255),
    date_loaded DATETIME DEFAULT CURRENT_TIMESTAMP,
    fpl_count INT
);

-- Tabla principal de mensajes FPL
CREATE TABLE IF NOT EXISTS flight_plans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    callsign VARCHAR(10) NOT NULL,
    departure_airport_id INT,
    arrival_airport_id INT,
    departure_time VARCHAR(5),
    arrival_time VARCHAR(5),
    raw_route TEXT,
	metadata_id INT,
    FOREIGN KEY (departure_airport_id) REFERENCES airports(id),
    FOREIGN KEY (arrival_airport_id) REFERENCES airports(id),
	FOREIGN KEY (metadata_id) REFERENCES metadata(id)
);

-- Tabla de puntos de ruta (waypoints)
CREATE TABLE IF NOT EXISTS waypoints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fpl_message_id INT,
    sequence INT,
    name VARCHAR(50),
    FOREIGN KEY (fpl_message_id) REFERENCES flight_plans(id)
);

